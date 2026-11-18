#!/usr/bin/env python3
"""Pure-offline horizontal residual-PCMD hold task, not a flight interface.

The existing reduced-order R5 plant and production hold/envelope methods are
reused. ``wind`` means residual ground drift, never atmospheric wind speed.
All truth is confined to the plant, reward and diagnostics. The observation
packer accepts only fused-style telemetry and software-approved commands.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np

TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
import sim_route_autoflight as sim

rpf = sim.rpf
CONTROL_DT_S = sim.DT
TRAIN_SCENARIOS = ("calm", "steady", "gust", "reversal", "dropout")
SCENARIOS = TRAIN_SCENARIOS + ("stress",)
FRAME_FIELDS = (
    "target_error_forward_over_servo_span", "target_error_right_over_servo_span",
    "velocity_forward_over_speed_limit", "velocity_right_over_speed_limit",
    "heading_error_sin", "heading_error_cos",
    "attitude_roll_over_max_tilt", "attitude_pitch_over_max_tilt",
    "baseline_roll_over_pcmd_cap", "baseline_pitch_over_pcmd_cap",
    "previous_approved_roll_over_pcmd_cap", "previous_approved_pitch_over_pcmd_cap",
    "previous_allocated_residual_roll_over_residual_cap",
    "previous_allocated_residual_pitch_over_residual_cap",
    "pose_capture_age_s", "attitude_event_age_s", "velocity_event_age_s",
    "pose_receipt_age_s", "attitude_receipt_age_s", "velocity_receipt_age_s",
    "pose_valid", "attitude_valid", "velocity_valid",
)


@dataclass(frozen=True)
class WindEnvConfig:
    episode_seconds: float = 30.0
    residual_cap_pct: float = 10.0
    history_steps: int = 8
    randomize: bool = True

    def __post_init__(self):
        if not math.isfinite(self.episode_seconds) or self.episode_seconds <= 0:
            raise ValueError("episode_seconds must be finite and positive")
        if not math.isfinite(self.residual_cap_pct) or not 0 <= self.residual_cap_pct <= 10:
            raise ValueError("residual_cap_pct must be between 0 and 10")
        if isinstance(self.history_steps, bool) or not isinstance(self.history_steps, int) or self.history_steps < 1:
            raise ValueError("history_steps must be a positive integer")
        if not isinstance(self.randomize, bool):
            raise ValueError("randomize must be boolean")


def pack_observation(frames: Sequence[Mapping[str, float]], history_steps: int = 8) -> np.ndarray:
    """Pack oldest-to-newest already-normalized deployable telemetry frames.

    Keys are exactly FRAME_FIELDS. Missing numeric values use zero, missing
    ages use one, and validity must be supplied explicitly (defaults to zero).
    Ages are host freshness except pose_capture_age_s. In particular, callers
    must not substitute unobservable SDK aircraft capture times for event ages.
    Missing history is zero padded with age=1 and masks=0, not duplicated.
    """
    if history_steps < 1:
        raise ValueError("history_steps must be positive")
    result = np.zeros((history_steps, len(FRAME_FIELDS)), dtype=np.float32)
    result[:, 14:20] = 1.0
    selected = frames[-history_steps:]
    start = history_steps - len(selected)
    for row, frame in enumerate(selected, start):
        for col, name in enumerate(FRAME_FIELDS):
            value = float(frame.get(name, 1.0 if 14 <= col < 20 else 0.0))
            if not math.isfinite(value):
                value = 1.0 if 14 <= col < 20 else 0.0
            result[row, col] = max(0.0 if col >= 14 else -5.0, min(1.0 if col >= 14 else 5.0, value))
    return result.reshape(-1)


class AnafiWindEnv(gym.Env):
    """20 Hz fixed-height/fixed-heading hold, bounded roll/pitch residual only."""
    metadata = {"render_modes": [], "render_fps": 20}

    def __init__(self, config: WindEnvConfig | None = None):
        super().__init__()
        self.config = config or WindEnvConfig()
        self.max_episode_steps = math.ceil(self.config.episode_seconds / CONTROL_DT_S)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.tile(np.array([-5.0] * 14 + [0.0] * 9, dtype=np.float32), self.config.history_steps)
        high = np.tile(np.array([5.0] * 14 + [1.0] * 9, dtype=np.float32), self.config.history_steps)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self._started = False
        self._done = False
        self.map_frame = rpf.MapFrame(np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                                      np.array([0., 0., 1.]), source="offline_horizontal_basis")

    def contract(self) -> dict:
        """JSON-serializable deployment schema and explicit simulation boundary."""
        return {
            "schema_version": 2, "task": "fixed_height_fixed_heading_horizontal_hold",
            "config": asdict(self.config), "control_dt_s": CONTROL_DT_S,
            "control_hz": 20, "plant_hz": 200, "max_episode_steps": self.max_episode_steps,
            "action": {"shape": [2], "dtype": "float32", "bounds": [-1., 1.],
                       "order": ["roll", "pitch"], "residual_cap_pct": self.config.residual_cap_pct,
                       "allocation": "nearest-integer residual clipped to integer cap, then baseline-priority production envelope; feedback excludes residual"},
            "observation": {"shape": list(self.observation_space.shape), "dtype": "float32",
                            "frame_fields": list(FRAME_FIELDS), "frame_size": len(FRAME_FIELDS),
                            "history_steps": self.config.history_steps, "order": "oldest_to_newest",
                            "numeric_clip": [-5., 5.], "age_clip_s": [0., 1.],
                            "missing": "zero numeric sentinel, age one, validity zero; zero padded missing history",
                            "packer": "anafi_wind_env.pack_observation",
                            "position_normalization": "map-unit body target error / max(configured slowdown distance, 4 arrival radii); no physical scale supplied to actor",
                            "velocity_normalization_mps": 0.8, "tilt_normalization_deg": 20.,
                            "pcmd_normalization_pct": 50.,
                            "privileged_excluded": ["true_position", "true_wind", "physical_capture_latency", "raw_imu", "thrust", "plant_parameters", "scenario"]},
            "clocks": {"pose_stamp": "visual source capture time",
                       "attitude_velocity_stamp": "host receipt/event time, not aircraft source capture time",
                       "capture_times": "internal simulation truth exposed only in diagnostics",
                       "cached_samples": "original source/event and receipt stamps preserved; never restamped",
                       "gate": "residual zero unless pose capture age<=0.5s, attitude receipt age<=0.5s, velocity receipt age<=0.5s"},
            "rates_hz": {"nominal": {"pose": 10., "attitude": 20., "velocity": 5.},
                         "randomized": {"pose": [8., 15.], "attitude": [10., 20.], "velocity": [4., 8.]},
                         "scheduler": "sample/receipt delivery quantized to 20Hz plant ticks"},
            "latency_s": {"nominal": {"pose": .12, "attitude": .04, "velocity": .20, "command": .025},
                          "randomized": {"pose": [.08, .24], "attitude": [.02, .08], "velocity": [.12, .35], "command": [0., .10]},
                          "stress": "pose0.22s, attitude0.08s, velocity0.32s, command0.08s plus longer dropout"},
            "model": {"nominal_label": "R5 fitted reduced-order residual-ground-drift plant",
                      "nominal": {"tau_h_s": 3., "tau_tilt_s": .4, "horizontal_accel_gain": 1.09},
                      "randomized_label": "unvalidated robustness range around R5; stress combination/profile withheld, individual slow/delayed components covered",
                      "randomized": {"tau_h_s": [2., 4.5], "tau_tilt_s": [.25, .75], "horizontal_accel_gain": [.9, 1.25]},
                      "hover_anchor": None, "hover_hold_mps": 0., "fixed_height_m": 2.,
                      "wind_model": "residual_ground_drift_proxy", "process_noise_sigma_mps": 0.,
                      "controller": "production YawAlignedPcmdController FINAL_HOLD with calibrated map-unit servo span and radius",
                      "envelope": "production _limit_acceleration_slew, _apply_speed_limit, _bank_envelope_limited_accel; software speed limit0.8m/s"},
            "scenarios": {"train": list(TRAIN_SCENARIOS), "heldout": ["stress"],
                          "calm": "exactly zero drift", "steady": "constant drift",
                          "gust": "finite triangular pulse with full post-pulse recovery window",
                          "reversal": "positive then negative residual drift, returns to zero",
                          "dropout": "steady drift, short pose/attitude/velocity event outages",
                          "stress": "held-out larger pulse/reversal, slower plant and delayed intermittent telemetry"},
            "reward": "negative true horizontal squared error +0.2*ground speed squared +0.005*requested residual/cap squared +0.02*unallocated integer residual/cap squared +0.001*residual change/cap squared, per control period; failure penalty25",
            "failure": {"horizontal_error_m": 8., "ground_speed_mps": 3., "nonfinite_state": True},
            "saturation_fraction": "fraction of steps with baseline envelope modification, allocated residual different from nearest-integer requested PCMD, or approved command at horizontal cap; fractional-to-integer quantization alone excluded",
            "recovery": {"definition": "elapsed from final finite drift-profile return-to-zero until first continuous1s interval with true horizontal error<=0.2m and ground speed<=0.1m/s; first interval start reported; episode continues through full horizon",
                         "null": "not_applicable for calm/steady/dropout; not_recovered if no interval; insufficient_post_gust_window if horizon/failure ends before1s window"},
            "diagnostic_keys": ["position_error_true_m", "position_error_estimated_u", "sample_stamps", "sample_ages", "sample_masks", "residual_enabled", "requested_residual_pct", "allocated_residual_pct", "baseline_pcmd", "baseline_requested_pcmd", "approved_pcmd", "transport_pcmd", "drift_mps", "episode_metrics"],
            "source_paths": [str(TOOLS_ROOT / "anafi_wind_env.py"), str(TOOLS_ROOT / "sim_route_autoflight.py"),
                             str(sim.FLIGHT_ROOT / "real_path_follow_controller.py"),
                             str(sim.OPERATOR_ROOT / "operator_autonomy.py")],
            "limitations": ["No hardware/SDK connection, Sphinx, takeoff, landing or transport execution.",
                            "Synthetic fused-style noisy map poses and delayed attitude/groundspeed, not the production camera/fusion pipeline.",
                            "Residual drift is not atmospheric wind; fitted acceleration/drag is not proprietary firmware or full aerodynamics.",
                            "Fixed yaw/height, no obstacles, raw IMU, thrust observations, battery/ground-effect changes or vertical disturbance.",
                            "Hover anchor and spring (including anchor-triggered zero-command braking) disabled; no perfect hidden position hold.",
                            "Approved PCMD is software authorization, not actuator readback; telemetry receipt age does not prove physical freshness.",
                            "Offline learned performance is not proof of real wind resistance or flight authorization."],
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32 - 1))
        self.episode_seed = int(seed)
        streams = np.random.SeedSequence(self.episode_seed).spawn(6)
        self._domain_rng, self._plant_rng, self._pose_rng, self._att_rng, self._vel_rng, self._command_rng = [np.random.default_rng(s) for s in streams]
        requested_scenario = (options or {}).get("scenario")
        self.scenario = str(requested_scenario) if requested_scenario is not None else str(self._domain_rng.choice(TRAIN_SCENARIOS))
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        self._configure_episode()
        self._step_count = 0
        # Positive host clock origin avoids ambiguous zero SDK receipt stamps.
        self._clock_origin = 1.
        self._now = self._clock_origin
        self._pending_samples = []
        self._event_counter = 0
        self._samples = {name: None for name in ("pose", "attitude", "velocity")}
        self._next_capture = {name: 0. for name in self._samples}
        self._command_queue = deque()
        self._transport_pcmd = (0, 0, 0, 0)
        self._previous_approved = (0, 0, 0, 0)
        self._previous_residual = np.zeros(2)
        self._history = deque(maxlen=self.config.history_steps)
        self._errors = []
        self._speeds = []
        self._residual_squared = 0.
        self._pcmd_peak = 0.
        self._saturated_steps = 0
        self._recovery_start = None
        self._recovery_time = None
        self._failure_reason = None
        self._gust_phase = 0.
        self._controller = rpf.YawAlignedPcmdController(self._cfg)
        self._guard = sim._SimulationAutonomyChecks(.8, SimpleNamespace(cfg=self._cfg))
        self._pump_samples()
        self._prepare_baseline()
        self._history.append(self._frame())
        self._started, self._done = True, False
        info = self._diagnostics(np.zeros(2), np.zeros(2), (0, 0, 0, 0), (0, 0, 0, 0))
        info["scenario"] = self.scenario
        info["seed"] = self.episode_seed
        info["model_parameters"] = self._model_parameters()
        return pack_observation(list(self._history), self.config.history_steps), info

    def _configure_episode(self):
        rng = self._domain_rng
        randomize = self.config.randomize
        self._physical_scale = float(rng.uniform(7., 13.)) if randomize else 10.
        self._reported_scale = self._physical_scale * (float(rng.uniform(.95, 1.05)) if randomize else 1.)
        self._heading = float(rng.uniform(-math.pi, math.pi)) if randomize else 0.
        offset = rng.uniform(-.35, .35, size=2) if randomize else np.array([.25, -.15])
        self._target_m = np.array([0., 0., 2.])
        self._state = sim.DroneState(self._target_m + np.array([offset[0], offset[1], 0.]), self._heading)
        # Authored physical hold radius/span converted using available scale calibration.
        self._cfg = sim.replace(rpf.production_auto_control_config(self.map_frame),
                                waypoint_arrive_radius=.2 / self._reported_scale,
                                slowdown_distance=1.5 / self._reported_scale,
                                translation_arrival_tolerance=.002 / self._reported_scale)
        self._target_u = self._target_m / self._physical_scale
        tau_h = float(rng.uniform(2., 4.5)) if randomize else 3.
        tau_tilt = float(rng.uniform(.25, .75)) if randomize else .4
        accel = float(rng.uniform(.9, 1.25)) if randomize else 1.09
        self._rates = {"pose": float(rng.uniform(8., 15.)) if randomize else 10.,
                       "attitude": float(rng.uniform(10., 20.)) if randomize else 20.,
                       "velocity": float(rng.uniform(4., 8.)) if randomize else 5.}
        self._latencies = {"pose": float(rng.uniform(.08, .24)) if randomize else .12,
                           "attitude": float(rng.uniform(.02, .08)) if randomize else .04,
                           "velocity": float(rng.uniform(.12, .35)) if randomize else .20}
        self._command_latency = float(rng.uniform(0., .10)) if randomize else .025
        direction = float(rng.uniform(-math.pi, math.pi))
        amplitude = float(rng.uniform(.18, .4)) if randomize else .3
        unit = np.array([math.cos(direction), math.sin(direction), 0.])
        duration = self.max_episode_steps * CONTROL_DT_S
        self._recovery_origin = None
        profile = ()
        mean = np.zeros(3)
        if self.scenario in ("steady", "dropout"):
            mean = amplitude * unit
        elif self.scenario == "gust":
            times, values = (0., .25, .33, .45, 1.), (0., 0., 1., 0., 0.)
            profile = tuple((t * duration, *(amplitude * 1.8 * v * unit)) for t, v in zip(times, values))
            self._recovery_origin = .45 * duration
        elif self.scenario in ("reversal", "stress"):
            times, values = (0., .15, .25, .38, .48, .58, 1.), (0., 0., 1., 1., -1., 0., 0.)
            gain = 2.5 if self.scenario == "stress" else 1.3
            profile = tuple((t * duration, *(amplitude * gain * v * unit)) for t, v in zip(times, values))
            self._recovery_origin = .58 * duration
        self._outages = {"pose": [], "attitude": [], "velocity": []}
        if self.scenario in ("dropout", "stress"):
            multiplier = 1.5 if self.scenario == "stress" else 1.
            for name, fraction, length in (("pose", .35, .85), ("attitude", .50, .65), ("velocity", .65, 1.15)):
                start = fraction * duration
                self._outages[name].append((start, start + min(length * multiplier, duration * .12)))
        if self.scenario == "stress":
            tau_h, tau_tilt, accel = 4.2, .65, .90
            self._latencies = {"pose": .22, "attitude": .08, "velocity": .32}
            self._command_latency = .08
        self._params = sim.SimParams(route=TOOLS_ROOT, align=None,
                                    seed=self.episode_seed, duration_s=duration,
                                    meters_per_unit=self._physical_scale,
                                    tau_h_s=tau_h, tau_tilt_s=tau_tilt,
                                    horizontal_accel_gain=accel, hover_hold_mps=0.,
                                    wind_mean_mps=tuple(float(v) for v in mean), wind_profile=profile,
                                    wind_sigma_mps=0., wind_tau_s=.6,
                                    speed_limit_mps=.8, gust_every_s=0., gust_m=0.)
        self._wind = mean.copy()

    def _model_parameters(self):
        return {"label": "heldout_unvalidated_stress" if self.scenario == "stress" else ("randomized_R5_perturbation" if self.config.randomize else "nominal_R5_fit"),
                "physical_meters_per_unit": self._physical_scale, "reported_meters_per_unit": self._reported_scale,
                "tau_h_s": self._params.tau_h_s, "tau_tilt_s": self._params.tau_tilt_s,
                "horizontal_accel_gain": self._params.horizontal_accel_gain,
                "rates_hz": dict(self._rates), "latency_s": dict(self._latencies),
                "command_latency_s": self._command_latency}

    def _elapsed(self):
        return self._step_count * CONTROL_DT_S

    def _pump_samples(self):
        elapsed = self._elapsed()
        for name, rng in (("pose", self._pose_rng), ("attitude", self._att_rng), ("velocity", self._vel_rng)):
            while self._next_capture[name] <= elapsed + 1e-10:
                capture = self._next_capture[name]
                self._next_capture[name] += 1. / self._rates[name]
                # Always consume event/noise randomness, even if the event is dropped.
                noise = rng.normal(size=3)
                random_drop = float(rng.random()) < (.01 if self.config.randomize else 0.)
                outage = any(start <= capture < end for start, end in self._outages[name])
                if name == "pose":
                    value = self._state.pos_m / self._physical_scale + noise * (.015 / self._physical_scale)
                    value[2] = self._target_u[2]
                    value = (*value, rpf.wrap_angle(self._state.yaw + noise[2] * math.radians(.25)))
                elif name == "attitude":
                    # Tilt-state-derived attitude, never synthetic raw IMU channels.
                    value = (rpf.wrap_angle(self._state.yaw + noise[0] * math.radians(.15)),
                             self._state.tilt_right + noise[1] * math.radians(.10),
                             -self._state.tilt_fwd + noise[2] * math.radians(.10))
                else:
                    ground = self._state.vel_m + self._wind + self._state.gust_velocity
                    value = (float(ground[0] + noise[0] * .015), float(ground[1] + noise[1] * .015))
                if not (random_drop or outage):
                    self._event_counter += 1
                    receipt = self._clock_origin + capture + self._latencies[name]
                    heapq.heappush(self._pending_samples, (receipt, self._event_counter, name,
                                                          self._clock_origin + capture, value))
        while self._pending_samples and self._pending_samples[0][0] <= self._now + 1e-10:
            _delivery, _order, name, capture, value = heapq.heappop(self._pending_samples)
            # Callback happens on this scheduler tick, not at the hidden physical capture.
            receipt = self._now
            stamp = capture if name == "pose" else receipt
            self._samples[name] = {"capture": capture, "receipt": receipt, "stamp": stamp, "value": value}
        self._guard.stamp = self._now
        state = self._guard.backend.state
        attitude = self._samples["attitude"]
        if attitude is None:
            state.att_yaw = float("nan")
            state.attitude_mono_ns = 0
        else:
            state.att_yaw = rpf.wrap_angle(math.pi / 2. - attitude["value"][0]) if self._valid("attitude") else float("nan")
            state.attitude_mono_ns = int(attitude["stamp"] * 1e9)
        velocity = self._samples["velocity"]
        if velocity is None:
            state.ground_speed_mps = float("nan")
            state.ground_speed_mono_ns = 0
            state.speed_north_mps = state.speed_east_mps = float("nan")
        else:
            east, north = velocity["value"]
            state.speed_east_mps, state.speed_north_mps = east, north
            state.ground_speed_mps = math.hypot(east, north)
            state.ground_speed_mono_ns = int(velocity["stamp"] * 1e9)

    def _valid(self, name):
        sample = self._samples[name]
        limit = float(self._cfg.max_pose_age_s)  # Production pose/body-velocity age bound: 0.5 s.
        return sample is not None and 0 <= self._now - sample["stamp"] <= limit + 1e-10

    def _prepare_baseline(self):
        self._residual_enabled = all(self._valid(name) for name in self._samples)
        sample = self._samples["pose"]
        self._pose = None
        if self._valid("pose"):
            x, y, z, yaw = sample["value"]
            if self._valid("attitude"):
                yaw = self._samples["attitude"]["value"][0]
            self._pose = rpf.Pose(float(x), float(y), float(z), float(yaw), stamp=sample["stamp"])
            command = rpf.Command("FINAL_HOLD", np.zeros(3), self._heading, self._target_u,
                                  0., 0., should_land=False)
            self._baseline_requested = self._controller.update(command, self._pose, self._now,
                                                               target_key=0, body_velocity=self._guard._body_velocity())
        else:
            self._controller.reset()
            self._baseline_requested = (0, 0, 0, 0)
        # Task fixes yaw/height. FINAL_HOLD already has zero yaw; zero gaz avoids
        # injecting height control into this explicitly horizontal experiment.
        self._baseline_requested = (self._baseline_requested[0], self._baseline_requested[1], 0, 0)

    def _guard_snapshot(self):
        accel = self._guard._accel_command
        return (None if accel is None else (accel[0], accel[1].copy()),
                self._guard._speed_sample, self._guard._speed_rate_mps2)

    def _guard_restore(self, snapshot):
        accel, sample, rate = snapshot
        self._guard._accel_command = None if accel is None else (accel[0], accel[1].copy())
        self._guard._speed_sample, self._guard._speed_rate_mps2 = sample, rate

    def _approve(self, pcmd):
        ramped = self._guard._limit_acceleration_slew(pcmd)
        approved = self._guard.apply(ramped, self._now)
        self._guard._bank_envelope_limited_accel(ramped, approved)
        return tuple(int(v) for v in approved)

    def _quantize_residual(self, requested):
        cap = math.floor(self.config.residual_cap_pct)
        return np.clip(np.rint(requested), -cap, cap).astype(int)

    def _allocate(self, requested):
        initial = self._guard_snapshot()
        baseline = self._approve(self._baseline_requested)
        baseline_state = self._guard_snapshot()
        approved = baseline
        allocated = np.zeros(2)
        quantized = self._quantize_residual(requested)
        if self._residual_enabled and np.any(quantized):
            cap = self._cfg.max_translation_pcmd
            # Accept only corrections which remain bounded relative to the
            # independently approved baseline after the SAME production envelope.
            # Rejected counterfactuals cannot bank slew or speed state.
            for scale in (1., .5, .25):
                delta = np.rint(quantized * scale).astype(int)
                candidate = (int(np.clip(self._baseline_requested[0] + delta[0], -cap, cap)),
                             int(np.clip(self._baseline_requested[1] + delta[1], -cap, cap)), 0, 0)
                self._guard_restore(initial)
                final = self._approve(candidate)
                actual = np.array(final[:2], dtype=float) - np.array(baseline[:2], dtype=float)
                same_direction = np.all(actual * quantized >= 0)
                bounded = np.all(np.abs(actual) <= np.abs(quantized))
                if same_direction and bounded:
                    approved, allocated = final, actual
                    break
            else:
                self._guard_restore(baseline_state)
        # Exact zero-residual path is one production approval; feedback always
        # refers to its baseline-attributed part, not baseline+residual.
        self._controller.feedback_applied(baseline)
        return baseline, approved, allocated

    def _frame(self):
        frame = dict.fromkeys(FRAME_FIELDS, 0.)
        for index, name in enumerate(("pose", "attitude", "velocity")):
            sample = self._samples[name]
            frame[FRAME_FIELDS[14 + index]] = min(1., self._now - sample["stamp"]) if sample else 1.
            frame[FRAME_FIELDS[17 + index]] = min(1., self._now - sample["receipt"]) if sample else 1.
            frame[FRAME_FIELDS[20 + index]] = float(self._valid(name))
        if self._pose is not None:
            delta = self._target_u - self._pose.xyz
            forward, right, _ = self.map_frame.body_components(delta, self._pose.yaw)
            span = max(self._cfg.slowdown_distance, 4. * self._cfg.waypoint_arrive_radius)
            frame[FRAME_FIELDS[0]], frame[FRAME_FIELDS[1]] = forward / span, right / span
        if self._valid("attitude"):
            yaw, roll, pitch = self._samples["attitude"]["value"]
            error = rpf.wrap_angle(self._heading - yaw)
            frame[FRAME_FIELDS[4]], frame[FRAME_FIELDS[5]] = math.sin(error), math.cos(error)
            frame[FRAME_FIELDS[6]], frame[FRAME_FIELDS[7]] = roll / math.radians(20.), pitch / math.radians(20.)
        velocity = self._guard._body_velocity()
        if velocity is not None and self._valid("attitude"):
            frame[FRAME_FIELDS[2]], frame[FRAME_FIELDS[3]] = velocity[0] / .8, velocity[1] / .8
        cap = float(self._cfg.max_translation_pcmd)
        for index in range(2):
            frame[FRAME_FIELDS[8 + index]] = self._baseline_requested[index] / cap
            frame[FRAME_FIELDS[10 + index]] = self._previous_approved[index] / cap
            frame[FRAME_FIELDS[12 + index]] = self._previous_residual[index] / max(self.config.residual_cap_pct, 1.)
        return frame

    def _diagnostics(self, requested, allocated, baseline, approved):
        estimated_error = None if self._pose is None else self.map_frame.horizontal_distance(self._target_u - self._pose.xyz)
        samples = self._samples
        return {"position_error_true_m": float(np.linalg.norm((self._state.pos_m - self._target_m)[:2])),
                "position_error_estimated_u": estimated_error,
                "sample_stamps": {name: None if sample is None else {"source_or_event_s": float(sample["stamp"]),
                                  "receipt_s": float(sample["receipt"]), "physical_capture_s": float(sample["capture"])} for name, sample in samples.items()},
                "sample_ages": {name: None if sample is None else {"source_or_event_s": float(self._now - sample["stamp"]),
                                "receipt_s": float(self._now - sample["receipt"])} for name, sample in samples.items()},
                "sample_masks": {name: bool(self._valid(name)) for name in samples},
                "residual_enabled": bool(self._residual_enabled),
                "requested_residual_pct": [float(v) for v in requested],
                "allocated_residual_pct": [float(v) for v in allocated],
                "baseline_pcmd": list(baseline), "baseline_requested_pcmd": list(self._baseline_requested),
                "approved_pcmd": list(approved), "transport_pcmd": list(self._transport_pcmd),
                "drift_mps": [float(v) for v in self._wind], "elapsed_s": self._elapsed()}

    def step(self, action):
        if not self._started or self._done:
            raise RuntimeError("reset() is required before stepping a new episode")
        values = np.asarray(action, dtype=float)
        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError("action must be two finite roll/pitch residual values")
        requested = np.clip(values, -1., 1.) * self.config.residual_cap_pct
        used_gate = self._residual_enabled
        used_baseline_requested = self._baseline_requested
        decision_info = self._diagnostics(requested, np.zeros(2), (0, 0, 0, 0), (0, 0, 0, 0))
        baseline, approved, allocated = self._allocate(requested)
        self._command_queue.append((self._now + self._command_latency, approved))
        while self._command_queue and self._command_queue[0][0] <= self._now + 1e-10:
            _, self._transport_pcmd = self._command_queue.popleft()
        self._wind, self._gust_phase, _ = sim._integrate_plant(
            self._state, self._wind, None, self._params, self.map_frame,
            self._plant_rng, self._transport_pcmd, self._params.cruise_mps / self._cfg.max_translation_pcmd,
            self._params.max_vertical_speed_mps / 100., self._params.max_rotation_speed_deg_s / 100.,
            self._gust_phase, elapsed_s=self._elapsed())
        self._step_count += 1
        self._now = self._clock_origin + self._elapsed()
        error = float(np.linalg.norm((self._state.pos_m - self._target_m)[:2]))
        speed = float(np.linalg.norm((self._state.vel_m + self._wind + self._state.gust_velocity)[:2]))
        finite = math.isfinite(error) and math.isfinite(speed) and np.all(np.isfinite(self._state.pos_m))
        if not finite:
            self._failure_reason = "nonfinite_state"
            error, speed = 8., 3.
        elif error > 8.:
            self._failure_reason = "horizontal_out_of_bounds"
        elif speed > 3.:
            self._failure_reason = "ground_speed_out_of_bounds"
        terminated = self._failure_reason is not None
        truncated = self._step_count >= self.max_episode_steps and not terminated
        self._done = terminated or truncated
        residual_scale = max(self.config.residual_cap_pct, 1.)
        unallocated = self._quantize_residual(requested) - allocated if used_gate else np.zeros(2)
        reward = -(error**2 + .2 * speed**2 + .005 * float(np.sum((requested / residual_scale)**2))
                   + .02 * float(np.sum((unallocated / residual_scale)**2))
                   + .001 * float(np.sum(((allocated - self._previous_residual) / residual_scale)**2))) * CONTROL_DT_S
        if terminated:
            reward -= 25.
        self._errors.append(error)
        self._speeds.append(speed)
        self._residual_squared += float(np.sum(allocated**2))
        self._pcmd_peak = max(self._pcmd_peak, float(max(abs(v) for v in approved[:2])))
        saturation = (baseline[:2] != used_baseline_requested[:2] or
                      (used_gate and not np.array_equal(allocated, self._quantize_residual(requested))) or
                      max(abs(v) for v in approved[:2]) >= self._cfg.max_translation_pcmd)
        self._saturated_steps += int(saturation)
        if self._recovery_origin is not None and self._elapsed() >= self._recovery_origin:
            if error <= .2 and speed <= .1:
                if self._recovery_start is None:
                    self._recovery_start = self._elapsed()
                if self._recovery_time is None and self._elapsed() - self._recovery_start >= 1. - 1e-10:
                    self._recovery_time = max(0., self._recovery_start - self._recovery_origin)
            else:
                self._recovery_start = None
        self._previous_approved, self._previous_residual = approved, allocated.copy()
        self._pump_samples()
        self._prepare_baseline()
        self._history.append(self._frame())
        info = self._diagnostics(requested, allocated, baseline, approved)
        # Main sample diagnostics correspond to the action's decision time.
        # next_* sample diagnostics describe the returned observation instead.
        for key in ("sample_stamps", "sample_ages", "sample_masks"):
            info["next_" + key] = info[key]
            info[key] = decision_info[key]
        info["residual_enabled"] = bool(used_gate)
        info["next_residual_enabled"] = bool(self._residual_enabled)
        info["baseline_requested_pcmd"] = list(used_baseline_requested)
        info["next_baseline_requested_pcmd"] = list(self._baseline_requested)
        info["saturated"] = bool(saturation)
        if self._done:
            info["episode_metrics"] = self._episode_metrics()
        return pack_observation(list(self._history), self.config.history_steps), float(reward), bool(terminated), bool(truncated), info

    def _episode_metrics(self):
        n = len(self._errors)
        if self._recovery_origin is None:
            recovery_status = "not_applicable"
        elif self._recovery_time is not None:
            recovery_status = "recovered"
        elif self._elapsed() - self._recovery_origin < 1.:
            recovery_status = "insufficient_post_gust_window"
        else:
            recovery_status = "not_recovered"
        return {"scenario": self.scenario, "seed": self.episode_seed,
                "duration_s": self._elapsed(), "horizontal_rmse_m": float(np.sqrt(np.mean(np.square(self._errors)))),
                "horizontal_p95_m": float(np.percentile(self._errors, 95)), "max_error_m": float(max(self._errors)),
                "max_ground_speed_mps": float(max(self._speeds)), "pcmd_peak_pct": self._pcmd_peak,
                "residual_rms_pct": math.sqrt(self._residual_squared / (2 * n)),
                "saturation_fraction": self._saturated_steps / n,
                "failure": self._failure_reason is not None, "failure_reason": self._failure_reason,
                "recovery_time_s": self._recovery_time, "recovery_status": recovery_status}
