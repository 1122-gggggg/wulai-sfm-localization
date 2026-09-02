"""In-process kinematic plant for testing the production route controller."""
from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np

from real_path_follow_controller import MapFrame, MissionRouteSnapshot, Pose


class SimulatedRoutePlant:
    """Translate production PCMD decisions into a fast, hardware-free pose."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.state = backend.state
        self.nudge_pct = 10
        self.pilot_sticks = False
        self._runtime_safety_action_latched = False
        self._lock = threading.Lock()
        self._map_frame: MapFrame | None = None
        self._speed_limit_enabled: bool | None = None
        self.finished = False

    @property
    def active(self) -> bool:
        return self._map_frame is not None and not self.finished

    def begin(self, snapshot: MissionRouteSnapshot, map_frame: MapFrame) -> bool:
        if not isinstance(map_frame, MapFrame) or len(snapshot.waypoints) < 2:
            return False
        snapshot.verify_file_unchanged()
        points = np.asarray(snapshot.waypoints, dtype=float)
        if points.shape[1:] != (3,) or not np.isfinite(points).all():
            return False
        heading = map_frame.heading(points[1] - points[0])
        with self._lock:
            self.backend.sim_xyz = points[0].copy()
            self.backend.sim_yaw = heading
            self._map_frame = map_frame
            self.finished = False
            self.pilot_sticks = False
            self.state.mode = "AUTO"
            self.state.control_owner = "SIM_AUTO"
            self.state.flight_state = "flying"
            self.state.tracker_state = "ROUTE_TEST"
            self.state.last_command = "start_auto:simulated_route"
            self._speed_limit_enabled = bool(
                self.state.autonomous_speed_limit_enabled
            )
            # Map units intentionally have no metres conversion. This run tests
            # route geometry and controller decisions, not the physical m/s guard.
            self.state.autonomous_speed_limit_enabled = False
        return True

    def pose(self) -> Pose | None:
        if not self.active:
            return None
        with self._lock:
            values = self.backend.sim_xyz.copy()
            yaw = float(self.backend.sim_yaw)
        return Pose(*values, yaw=yaw, stamp=time.monotonic())

    def send_pcmd(
        self,
        roll: int,
        pitch: int,
        yaw: int,
        gaz: int,
        *,
        reason: str,
    ) -> bool:
        values = np.asarray((roll, pitch, yaw, gaz), dtype=float)
        if not np.isfinite(values).all():
            return False
        values = np.clip(values, -100.0, 100.0)
        map_frame = self._map_frame
        if map_frame is None:
            return not np.any(values)

        roll_value, pitch_value, yaw_value, gaz_value = values
        control_dt = 1.0 / 20.0
        # Translation runs four times faster than wall time so route checks stay
        # short. Yaw keeps the dry-run rate because its stability gate rejects jumps.
        translation_dt = 4.0 * control_dt
        with self._lock:
            self.backend.sim_yaw = (
                self.backend.sim_yaw + 0.06 * yaw_value * control_dt + math.pi
            ) % (2.0 * math.pi) - math.pi
            forward = (
                math.cos(self.backend.sim_yaw) * map_frame.east
                + math.sin(self.backend.sim_yaw) * map_frame.north
            )
            right = (
                math.sin(self.backend.sim_yaw) * map_frame.east
                - math.cos(self.backend.sim_yaw) * map_frame.north
            )
            displacement = (
                0.06 * pitch_value * translation_dt * forward
                + 0.06 * roll_value * translation_dt * right
                + 0.05 * gaz_value * translation_dt * map_frame.up
            )
            self.backend.sim_xyz += displacement
            self.state.pose[:] = [*self.backend.sim_xyz, self.backend.sim_yaw]
            self.state.att_yaw = self.backend.sim_yaw
            self.state.ground_speed_mps = (
                map_frame.horizontal_distance(displacement) / translation_dt
            )
            stamp = time.monotonic_ns()
            self.state.ground_speed_mono_ns = stamp
            self.state.telemetry_read_mono_ns = stamp
            self.state.last_pcmd_call_mono_ns = stamp
            self.state.last_command = reason
        return True

    def set_nudge_vector(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        gaz: float,
    ) -> bool:
        axes = np.asarray((roll, pitch, yaw, gaz), dtype=float)
        if not np.isfinite(axes).all():
            return False
        pcmd = np.rint(np.clip(axes, -1.0, 1.0) * self.nudge_pct).astype(int)
        return self.send_pcmd(*pcmd, reason="simulated_route_controller")

    def clear_nudge_vector(self) -> None:
        self.send_pcmd(0, 0, 0, 0, reason="simulated_route_hover")

    def give_to_pilot(self, *, reason: str = "manual") -> bool:
        self.clear_nudge_vector()
        self.pilot_sticks = True
        self.state.mode = "MANUAL"
        self.state.control_owner = "SIM_MANUAL"
        self.state.last_command = reason
        return True

    def finish(self) -> bool:
        self.clear_nudge_vector()
        with self._lock:
            self.finished = True
            self.state.mode = "MANUAL"
            self.state.control_owner = "SIM"
            self.state.flight_state = "landed"
            self.state.tracker_state = "ROUTE_TEST_COMPLETE"
            self.state.last_command = "simulated_route_complete"
            if self._speed_limit_enabled is not None:
                self.state.autonomous_speed_limit_enabled = self._speed_limit_enabled
            self._speed_limit_enabled = None
        return True


__all__ = ["SimulatedRoutePlant"]
