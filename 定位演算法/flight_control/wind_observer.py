"""Outer-loop residual ground-drift observer in NED metres per second.

Uses only authorized PCMD and firmware-fused NED velocity, not raw IMU or
actuator thrust. Inspired by Kamel 2017 augmented-disturbance estimation and
the shared outer-loop core of ESO/HGDO; firmware already compensates for wind,
so this estimates only the remaining ground drift. The flight loop logs the
estimate and its would-be feedforward; it never adds them to PCMD.
"""

from collections import deque
import math

import numpy as np

PCMD_SPEED_GAIN_MPS_PER_PCT = 0.04
RESPONSE_TAU_S = 1.6
RESPONSE_DELAY_S = 0.5
OBSERVER_TAU_S = 1.5
WARMUP_S = 3.0
MAX_SAMPLE_GAP_S = 0.5
MAX_DRIFT_MPS = 1.0
FEEDFORWARD_CAP_FRACTION = 0.3
COMMAND_HISTORY_S = 6.0


class ResidualDriftObserver:
    def __init__(self):
        self.last_reset = None
        self.reset()

    def reset(self) -> None:
        self.commands = deque()
        self.model = None
        self.drift = np.zeros(2)
        self.last_stamp = None
        self.started = None
        self.innovation = None
        self.samples = 0

    def observe_command(self, pcmd, yaw_ned, stamp) -> None:
        if yaw_ned is None or not math.isfinite(yaw_ned):
            self.reset()
            self.last_reset = "yaw_unavailable"
            return
        if self.commands and stamp < self.commands[-1][0]:
            return
        forward, right = pcmd[1], pcmd[0]
        c, s = math.cos(yaw_ned), math.sin(yaw_ned)
        self.commands.append((stamp, c * forward - s * right, s * forward + c * right))
        while len(self.commands) > 1 and self.commands[1][0] < stamp - COMMAND_HISTORY_S:
            self.commands.popleft()

    def observe_velocities(self, rows) -> None:
        for stamp, north, east in rows:
            if (not all(math.isfinite(v) for v in (stamp, north, east))
                    or self.last_stamp is not None and stamp <= self.last_stamp):
                continue
            self._update(stamp, np.array([north, east]))

    def _advance_model(self, stamp):
        start = self.last_stamp
        command = (0.0, 0.0)
        for sent, north, east in self.commands:
            change = sent + RESPONSE_DELAY_S
            if change <= start:
                command = (north, east)
                continue
            if change >= stamp:
                break
            self._integrate(change - start, command)
            start = change
            command = (north, east)
        self._integrate(stamp - start, command)

    def _integrate(self, dt, command):
        alpha = 1 - math.exp(-dt / RESPONSE_TAU_S)
        self.model[0] += (PCMD_SPEED_GAIN_MPS_PER_PCT * command[0] - self.model[0]) * alpha
        self.model[1] += (PCMD_SPEED_GAIN_MPS_PER_PCT * command[1] - self.model[1]) * alpha

    def _update(self, stamp, measured):
        if self.last_stamp is None or stamp - self.last_stamp > MAX_SAMPLE_GAP_S:
            self.model = measured
            self.drift[:] = 0
            self.innovation = None
            self.started = self.last_stamp = stamp
            self.samples = 1
            return
        dt = stamp - self.last_stamp
        self._advance_model(stamp)
        self.innovation = measured - self.model - self.drift
        self.drift += (1 - math.exp(-dt / OBSERVER_TAU_S)) * self.innovation
        self.samples += 1
        self.last_stamp = stamp
        if float(np.linalg.norm(self.drift)) > MAX_DRIFT_MPS:
            self.reset()
            self.last_reset = "diverged"

    def feedforward(self, now, yaw_ned, cap) -> tuple[tuple[float, float] | None, str]:
        if self.last_stamp is None:
            return None, "no_velocity"
        if now - self.last_stamp > MAX_SAMPLE_GAP_S:
            return None, "velocity_stale"
        if self.last_stamp - self.started < WARMUP_S:
            return None, "warming_up"
        if yaw_ned is None or not math.isfinite(yaw_ned):
            return None, "yaw_unavailable"
        north, east = (float(v) for v in -self.drift / PCMD_SPEED_GAIN_MPS_PER_PCT)
        magnitude = math.hypot(north, east)
        limit = FEEDFORWARD_CAP_FRACTION * cap
        if magnitude > limit:
            north *= limit / magnitude
            east *= limit / magnitude
        c, s = math.cos(yaw_ned), math.sin(yaw_ned)
        return (c * north + s * east, -s * north + c * east), "ready"

    def diagnostics(self) -> dict:
        return {"drift_ned_mps": [round(float(v), 4) for v in self.drift],
                "innovation_ned_mps": None if self.innovation is None else
                [round(float(v), 4) for v in self.innovation],
                "samples": self.samples,
                "warm_s": round(0.0 if self.started is None or self.last_stamp is None else
                                self.last_stamp - self.started, 4),
                "last_reset": self.last_reset}
