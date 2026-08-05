"""Lifecycle helpers for a local Parrot Sphinx instance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self, TextIO

from .models import TruePosition
from .route import Waypoint

ANAFI_DRONE_MODEL = "/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone"
ANAFI_ZERO_YAW_POSE = "0 0 0.2 0 0 0"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIRMWARE_CACHE = PROJECT_ROOT / "firmware" / "anafi-pc.ext2.zip"
DEFAULT_FIRMWARE_MANIFEST = PROJECT_ROOT / "firmware" / "manifest.json"


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    secure_boot_state: str
    firmware_source: str | None
    firmware_sha256: str | None
    failures: tuple[str, ...]


class SphinxPreflightError(RuntimeError):
    """Raised before launching a simulator that cannot run correctly."""


class FirmwareSourceError(RuntimeError):
    """Raised when no verified local ANAFI PC firmware image is available."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_default_firmware(path: Path) -> None:
    try:
        manifest = json.loads(DEFAULT_FIRMWARE_MANIFEST.read_text(encoding="utf-8"))
        expected_name = str(manifest["artifact"])
        expected_bytes = int(manifest["bytes"])
        expected_sha256 = str(manifest["sha256"]).lower()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise FirmwareSourceError(
            f"invalid ANAFI firmware manifest: {DEFAULT_FIRMWARE_MANIFEST}"
        ) from error
    if path.name != expected_name:
        raise FirmwareSourceError(f"firmware manifest expects {expected_name}, got {path.name}")
    if path.stat().st_size != expected_bytes:
        raise FirmwareSourceError(f"firmware size does not match manifest: {path}")
    if _sha256(path) != expected_sha256:
        raise FirmwareSourceError(f"firmware SHA-256 does not match manifest: {path}")


def _secure_boot_state() -> str:
    if shutil.which("mokutil") is None:
        return "unknown (mokutil unavailable)"
    result = subprocess.run(["mokutil", "--sb-state"], check=False, capture_output=True, text=True)
    text = f"{result.stdout}\n{result.stderr}".lower()
    if "secureboot enabled" in text:
        return "enabled"
    if "secureboot disabled" in text:
        return "disabled"
    return "unknown"


def run_preflight(firmware_source: str | Path | None = None) -> PreflightResult:
    """Validate Sphinx prerequisites without launching any simulator process."""
    failures: list[str] = []
    resolved_firmware: str | None = None
    firmware_sha256: str | None = None
    for executable in ("sphinx", "sphinx-cli", "parrot-ue4-empty", "tlm-data-logger"):
        if shutil.which(executable) is None:
            failures.append(f"required executable is missing: {executable}")
    if not Path(ANAFI_DRONE_MODEL).is_file():
        failures.append(f"ANAFI Sphinx model is missing: {ANAFI_DRONE_MODEL}")
    try:
        resolved_firmware = resolve_firmware_source(firmware_source)
        firmware_sha256 = _sha256(Path(resolved_firmware))
    except (FirmwareSourceError, OSError) as error:
        failures.append(str(error))

    secure_boot_state = _secure_boot_state()
    if secure_boot_state != "disabled":
        failures.append(
            "UEFI Secure Boot must be disabled for Parrot Sphinx firmware emulation "
            f"(detected: {secure_boot_state})"
        )
    return PreflightResult(
        passed=not failures,
        secure_boot_state=secure_boot_state,
        firmware_source=resolved_firmware,
        firmware_sha256=firmware_sha256,
        failures=tuple(failures),
    )


def require_preflight(firmware_source: str | Path | None = None) -> None:
    result = run_preflight(firmware_source)
    if not result.passed:
        details = "\n- ".join(result.failures)
        raise SphinxPreflightError(f"Sphinx preflight failed:\n- {details}")


def resolve_firmware_source(firmware_source: str | Path | None = None) -> str:
    """Resolve a local firmware image and verify the project-default cache."""
    candidate = (
        Path(firmware_source).expanduser().resolve()
        if firmware_source is not None
        else DEFAULT_FIRMWARE_CACHE.resolve()
    )
    if not candidate.is_file():
        raise FirmwareSourceError(
            f"verified local ANAFI PC firmware is required; missing file: {candidate}"
        )
    if candidate == DEFAULT_FIRMWARE_CACHE.resolve():
        _verify_default_firmware(candidate)
    return str(candidate)


def build_sphinx_command(
    output_dir: Path,
    *,
    firmware_source: str | Path | None = None,
    spawn_pose: str = ANAFI_ZERO_YAW_POSE,
    disable_front_camera: bool = False,
) -> list[str]:
    """Build a shell-free Sphinx launch command for the ANAFI PC firmware."""
    source = resolve_firmware_source(firmware_source)
    model_spec = f"{ANAFI_DRONE_MODEL}::firmware={source}::pose={spawn_pose}"
    if disable_front_camera:
        model_spec += "::with_front_cam=0"
    return [
        "sphinx",
        f"--datalog-outdir={output_dir}",
        "--datalog-rate=50",
        model_spec,
    ]


def build_ue_command(*, show_window: bool = False) -> list[str]:
    """Build the installed minimal UE world command, optionally with its window."""
    command = ["parrot-ue4-empty"]
    if show_window:
        command.append("-quality=low")
    else:
        command.append("-RenderOffScreen")
    return command


_SPHINX_PYTHON = "/usr/bin/python3.10"
_SPHINX_PYTHON_PATHS = (
    "/opt/parrot-sphinx/usr/lib/python3.10/site-packages",
    "/opt/parrot-sphinx/usr/lib/python/site-packages",
)
_MOVE_DRONE_SCRIPT = """
import sys
from pysphinx import Sphinx

dx, dy, dz = map(float, sys.argv[1:4])
sphinx = Sphinx()
machine = sphinx.get_default_machine_name()
ok = machine is not None and sphinx.move_drone(
    machine,
    (dx, dy, dz, 0.0, 0.0, 0.0),
    pose_ref=("val", "val", "val", "val", "val", "val"),
    pose_duration=0.05,
)
print("true" if ok else "false")
raise SystemExit(0 if ok else 1)
"""


def _move_drone_relative(dx_m: float, dy_m: float, dz_m: float = 0.0) -> None:
    """Move Sphinx true pose by a bounded relative offset."""
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            (
                "/opt/parrot-sphinx/usr/lib",
                environment.get("LD_LIBRARY_PATH", ""),
            ),
        )
    )
    environment["PYTHONPATH"] = ":".join(
        filter(
            None,
            (*_SPHINX_PYTHON_PATHS, environment.get("PYTHONPATH", "")),
        )
    )
    result = subprocess.run(
        [
            _SPHINX_PYTHON,
            "-c",
            _MOVE_DRONE_SCRIPT,
            str(dx_m),
            str(dy_m),
            str(dz_m),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"unable to apply bounded Sphinx displacement: {details}")


class SphinxWindDisplacements:
    """Apply seeded horizontal true-position offsets that model sudden wind drift."""

    def __init__(
        self,
        *,
        seed: int,
        maximum_displacement_m: float,
        interval_s: float,
    ) -> None:
        self._rng = random.Random(seed)
        self._maximum_displacement_m = maximum_displacement_m
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._started_at: float | None = None
        self.events: list[dict[str, float]] = []

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(
            target=self._run,
            name="sphinx-wind-displacements",
            daemon=True,
        )
        self._thread.start()

    def _apply_once(self) -> None:
        direction_rad = self._rng.uniform(-math.pi, math.pi)
        displacement_m = self._maximum_displacement_m * math.sqrt(self._rng.random())
        dx_m = displacement_m * math.cos(direction_rad)
        dy_m = displacement_m * math.sin(direction_rad)
        _move_drone_relative(dx_m, dy_m)
        self.events.append(
            {
                "elapsed_s": time.monotonic() - (self._started_at or time.monotonic()),
                "dx_m": dx_m,
                "dy_m": dy_m,
                "dz_m": 0.0,
                "displacement_m": displacement_m,
            }
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self._interval_s):
                self._apply_once()
        except BaseException as error:  # noqa: BLE001 -- surfaced by stop().
            self._error = error

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError("Sphinx wind-displacement worker failed") from self._error

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop()


class SphinxFinalApproachDisplacement:
    """Apply one maximum horizontal displacement as the final waypoint is entered."""

    def __init__(self, *, displacement_m: float, trigger_radius_m: float = 0.55) -> None:
        if displacement_m < 0.0 or trigger_radius_m <= 0.0:
            raise ValueError("displacement and trigger radius must be positive")
        self.displacement_m = displacement_m
        self.trigger_radius_m = trigger_radius_m
        self.applied = False
        self.events: list[dict[str, float]] = []
        self._started_at = time.monotonic()

    def maybe_apply(
        self,
        position: TruePosition,
        target: Waypoint,
        is_final_waypoint: bool,
    ) -> None:
        if self.applied or not is_final_waypoint or self.displacement_m == 0.0:
            return
        distance_m = math.dist(
            (position.x_m, position.y_m, position.z_m),
            (target.x_m, target.y_m, target.z_m),
        )
        if distance_m > self.trigger_radius_m:
            return
        away_x = position.x_m - target.x_m
        away_y = position.y_m - target.y_m
        horizontal = math.hypot(away_x, away_y)
        if horizontal <= 1e-9:
            away_x, away_y, horizontal = 0.0, 1.0, 1.0
        dx_m = self.displacement_m * away_x / horizontal
        dy_m = self.displacement_m * away_y / horizontal
        _move_drone_relative(dx_m, dy_m)
        self.applied = True
        self.events.append(
            {
                "elapsed_s": time.monotonic() - self._started_at,
                "dx_m": dx_m,
                "dy_m": dy_m,
                "dz_m": 0.0,
                "displacement_m": self.displacement_m,
            }
        )

    def stop(self) -> None:
        """Match the periodic disturbance cleanup interface."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop()


class SphinxInstance:
    """Own and cleanly tear down a Sphinx + UE process pair."""

    def __init__(
        self,
        output_dir: Path,
        *,
        firmware_source: str | Path | None = None,
        show_window: bool = False,
        spawn_pose: str = ANAFI_ZERO_YAW_POSE,
        disable_front_camera: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.firmware_source = firmware_source
        self.show_window = show_window
        self.spawn_pose = spawn_pose
        self.disable_front_camera = disable_front_camera
        self._sphinx: subprocess.Popen[str] | None = None
        self._ue: subprocess.Popen[str] | None = None
        self._log_handles: list[TextIO] = []

    def start(self, *, ready_timeout_s: float = 60.0) -> None:
        require_preflight(self.firmware_source)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sphinx_log = (self.output_dir / "sphinx.log").open("w", encoding="utf-8")
        ue_log = (self.output_dir / "unreal.log").open("w", encoding="utf-8")
        self._log_handles = [sphinx_log, ue_log]
        try:
            self._sphinx = subprocess.Popen(
                build_sphinx_command(
                    self.output_dir,
                    firmware_source=self.firmware_source,
                    spawn_pose=self.spawn_pose,
                    disable_front_camera=self.disable_front_camera,
                ),
                stdout=sphinx_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            time.sleep(1.0)
            self._ue = subprocess.Popen(
                build_ue_command(show_window=self.show_window),
                stdout=ue_log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            self._wait_until_ready(ready_timeout_s)
        except BaseException:
            self.stop()
            raise

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._sphinx is None or self._ue is None:
                raise RuntimeError("Sphinx process pair was not created")
            if self._sphinx.poll() is not None:
                raise RuntimeError("Sphinx exited before becoming ready; see sphinx.log")
            if self._ue.poll() is not None:
                raise RuntimeError(
                    "UE application exited before Sphinx became ready; see unreal.log"
                )
            result = subprocess.run(
                ["sphinx-cli", "info"], check=False, capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return
            time.sleep(1.0)
        raise TimeoutError("Sphinx did not become ready before the timeout; see simulator logs")

    def stop(self) -> None:
        for process in (self._ue, self._sphinx):
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        self._ue = None
        self._sphinx = None
        for handle in self._log_handles:
            handle.close()
        self._log_handles = []

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.stop()
