#!/usr/bin/env python3
"""
ESEKF — standalone Error-State EKF 15-dim for ANAFI NED velocity + visual pose fusion.

Coordinate conventions (documented header):
  world = GLOMAP map
  p  : camera center in map units  (3,)  ``p_cam = R * p_map + t``  (world->camera)
  v  : velocity in map units/s    (3,)  world NED metric converted via metres_per_map_unit
  q  : R_C_M  world->camera  wxyz  active  right-handed  normalized  w>0
  ba : accel bias  (3,)  body/camera frame
  bg : gyro bias   (3,)  body/camera frame
  T_C_B coincident  (camera == body)  — no extrinsic
  NED velocity is world NED m/s  (Parrot Olympe SpeedChanged already world)
     if metres_per_map_unit is not None:  v_map = v_ned / metres_per_map_unit
     else treat v_ned as already map units (caller ensures scale)

Nominal state:
  x = [p(3), v(3), q(wxyz 4), ba(3), bg(3)]  stored as p=np(3), v=np(3), q=np(4), ba=np(3), bg=np(3)
  timestamp last_predict

Error state 15-dim:
  delta = [dp(3), dv(3), dtheta(3), dba(3), dbg(3)]   covariance P 15x15

Dependencies: numpy, math only. No torch / GPU. O(15^3) via 6x6 inv + 15x15 muls.

Refs: standard ESEKF — dp_dot=dv, dv_dot=-R*ba, dtheta_dot=-R*bg, bias random walk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------------------
# quaternion helpers  (wxyz, right-handed active, q <- Exp(dtheta) * q)
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _normalize_q(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q = _normalize_q(q)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_q(q)
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=float,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def _exp_quat(dtheta: np.ndarray) -> np.ndarray:
    dtheta = np.asarray(dtheta, dtype=float).reshape(3)
    theta = float(np.linalg.norm(dtheta))
    if theta < 1e-8:
        # first-order: q ~ [1, 0.5*dtheta]
        w = 1.0 - theta * theta / 8.0
        vec = 0.5 * dtheta
        q = np.concatenate([[w], vec])
        return _normalize_q(q)
    half = theta * 0.5
    w = math.cos(half)
    s = math.sin(half) / theta
    vec = s * dtheta
    return _normalize_q(np.concatenate([[w], vec]))


def _log_quat(q_err: np.ndarray) -> np.ndarray:
    """2*log(q_err) -> rotation vector (3,). q_err wxyz expected small angle."""
    q_err = _normalize_q(q_err)
    # ensure shortest path (w>=0 already via normalize)
    w = float(np.clip(q_err[0], -1.0, 1.0))
    vec = q_err[1:].astype(float)
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        # small angle: 2*vec  (since vec ~ theta/2 * axis)
        return 2.0 * vec
    # general: angle = 2*atan2(n, w)
    angle = 2.0 * math.atan2(n, w)
    # clamp angle to [-pi, pi] via normalization already w>=0 => angle in [0,pi]
    axis = vec / n
    return axis * angle


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class EKFConfig:
    # init covariance diag
    pos_var: float = 0.1
    vel_var: float = 0.1
    att_var: float = 0.0076  # (5deg)^2 rad^2
    ba_var: float = 1e-2
    bg_var: float = 1e-4
    # continuous noise densities (variance per second)
    acc_noise: float = 0.10
    gyro_noise: float = 0.01
    ba_rw: float = 1e-4
    bg_rw: float = 1e-5
    # gating
    gate_threshold: float = 12.59  # chi2 95% 6DoF
    # timing / scale
    max_dt: float = 0.5
    metres_per_map_unit: float | None = None
    # prediction_allowed thresholds
    pos_trace_threshold: float = 1.0
    yaw_sigma_threshold_deg: float = 15.0
    max_age_frames: int = 6
    # world gravity (map world, NED down positive ~9.81). Set 0 to disable
    g_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # adaptive base covariances (variance)
    base_R_pos: float = 0.3
    base_R_yaw_deg: float = 5.0


# ---------------------------------------------------------------------------
# adaptive R helper (standalone + method)
# ---------------------------------------------------------------------------

def adaptive_visual_covariance(
    inliers: int | float | None = None,
    reproj_rmse: float | None = None,
    corr_count: int | float | None = None,
    fb_error: float | None = None,
    jump: float | np.ndarray | None = None,
    base_R_pos: float | np.ndarray = 0.3,
    base_R_yaw: float | None = None,
    base_R_yaw_deg: float | None = None,
    base_R_ori: float | np.ndarray | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return scaled (R_pos 3x3, R_ori 3x3).

    High quality: inliers>80 and reproj<1.5  -> *0.5
    Low quality:  inliers<30 or reproj>3     -> *2
    Also scales with corr_count, fb_error, jump if provided.
    """
    # alias handling
    if base_R_yaw_deg is None and base_R_yaw is not None:
        # if base_R_yaw looks like degrees (>0.5 rad ~28 deg) treat as deg
        # but spec says base_R_yaw=5deg, so assume degrees
        base_R_yaw_deg = float(base_R_yaw)
    if base_R_yaw_deg is None:
        base_R_yaw_deg = 5.0
    # allow caller to pass base_R_yaw_deg via kwargs base_R_yaw
    if "base_R_yaw" in kwargs and base_R_yaw_deg == 5.0 and base_R_yaw is None:
        try:
            base_R_yaw_deg = float(kwargs["base_R_yaw"])
        except Exception:
            pass

    # build base covariances
    if isinstance(base_R_pos, np.ndarray):
        arr = np.asarray(base_R_pos, dtype=float)
        if arr.shape == (3, 3):
            R_pos_base = arr.copy()
        elif arr.size == 3:
            R_pos_base = np.diag(arr.reshape(3).astype(float))
        elif arr.size == 1:
            R_pos_base = np.eye(3) * float(arr.ravel()[0])
        else:
            R_pos_base = np.eye(3) * 0.3
    else:
        try:
            v = float(base_R_pos)  # type: ignore[arg-type]
        except Exception:
            v = 0.3
        # variance directly (consistent with FusionConfig); not squared
        R_pos_base = np.eye(3, dtype=float) * v

    # yaw/orientation base variance from degrees
    if base_R_ori is not None:
        if isinstance(base_R_ori, np.ndarray):
            arr = np.asarray(base_R_ori, dtype=float)
            if arr.shape == (3, 3):
                R_ori_base = arr.copy()
            elif arr.size == 3:
                R_ori_base = np.diag(arr.reshape(3).astype(float))
            elif arr.size == 1:
                R_ori_base = np.eye(3) * float(arr.ravel()[0])
            else:
                R_ori_base = np.eye(3) * ((math.radians(float(base_R_yaw_deg))) ** 2)
        else:
            try:
                v = float(base_R_ori)  # type: ignore[arg-type]
                R_ori_base = np.eye(3, dtype=float) * v
            except Exception:
                R_ori_base = np.eye(3, dtype=float) * ((math.radians(float(base_R_yaw_deg))) ** 2)
    else:
        yaw_var = (math.radians(float(base_R_yaw_deg))) ** 2
        R_ori_base = np.eye(3, dtype=float) * yaw_var

    scale = 1.0

    # inliers / reproj primary gate
    try:
        if inliers is not None and reproj_rmse is not None:
            inl = float(inliers)
            rmse = float(reproj_rmse)
            if math.isfinite(inl) and math.isfinite(rmse):
                if inl >= 80 and rmse < 1.5:
                    scale *= 0.5
                elif inl < 30 or rmse > 3.0:
                    scale *= 2.0
                elif inl < 50 or rmse > 2.0:
                    scale *= 1.5
        elif inliers is not None:
            inl = float(inliers)
            if math.isfinite(inl):
                if inl >= 80:
                    scale *= 0.7
                elif inl < 30:
                    scale *= 2.0
        elif reproj_rmse is not None:
            rmse = float(reproj_rmse)
            if math.isfinite(rmse):
                if rmse < 1.5:
                    scale *= 0.7
                elif rmse > 3.0:
                    scale *= 2.0
    except Exception:
        pass

    # corr_count: fewer correspondences -> larger uncertainty
    try:
        if corr_count is not None:
            cc = float(corr_count)
            if math.isfinite(cc):
                if cc < 20:
                    scale *= 1.8
                elif cc < 40:
                    scale *= 1.3
                elif cc > 100:
                    scale *= 0.9
    except Exception:
        pass

    # fb_error: large forward-backward flow error -> larger R
    try:
        if fb_error is not None:
            fe = float(np.asarray(fb_error).ravel()[0]) if isinstance(fb_error, np.ndarray) else float(fb_error)
            if math.isfinite(fe):
                if fe > 1.0:
                    scale *= 1.8
                elif fe > 0.5:
                    scale *= 1.3
                elif fe < 0.2:
                    scale *= 0.95
    except Exception:
        pass

    # jump: large center jump between frames -> larger R
    try:
        if jump is not None:
            if isinstance(jump, np.ndarray):
                jv = float(np.linalg.norm(np.asarray(jump, dtype=float).reshape(-1)))
            else:
                jv = float(jump)
            if math.isfinite(jv):
                if jv > 2.0:
                    scale *= 2.0
                elif jv > 0.8:
                    scale *= 1.5
                elif jv < 0.2:
                    scale *= 0.95
    except Exception:
        pass

    # clamp scale to reasonable bounds
    scale = float(np.clip(scale, 0.25, 4.0))

    return R_pos_base * scale, R_ori_base * scale


def _adaptive_R_6x6(
    inliers=None, reproj_rmse=None, corr_count=None, fb_error=None, jump=None, base_R_pos=0.3, base_R_yaw_deg=5.0, **kw
) -> np.ndarray:
    Rp, Ro = adaptive_visual_covariance(
        inliers=inliers,
        reproj_rmse=reproj_rmse,
        corr_count=corr_count,
        fb_error=fb_error,
        jump=jump,
        base_R_pos=base_R_pos,
        base_R_yaw_deg=base_R_yaw_deg,
        **kw,
    )
    R = np.zeros((6, 6), dtype=float)
    R[0:3, 0:3] = Rp
    R[3:6, 3:6] = Ro
    return R


# ---------------------------------------------------------------------------
# ESEKF
# ---------------------------------------------------------------------------

class ESEKF:
    """
    Error-State EKF 15-dim.

    Nominal state: p, v, q(wxyz), ba, bg, last_predict timestamp
    Error: dp,dv,dtheta,dba,dbg  (15)

    Usage:
        ekf = ESEKF(EKFConfig(...))
        ekf.reset_from_pose(p, q_wxyz, v, timestamp)
        p_pred, q_pred, P, dtheta = ekf.predict(timestamp, velocity_ned=(vx,vy,vz))
        ok, d2, nu = ekf.update_visual(z_p, z_q, R_pos, R_ori, timestamp)
        ekf.get_state(); ekf.get_covariance(); ekf.prediction_allowed(); ekf.as_info()
    """

    def __init__(
        self,
        config: EKFConfig | None = None,
        metres_per_map_unit: float | None = None,
        gate_threshold: float | None = None,
        **kwargs,
    ):
        cfg = config if config is not None else EKFConfig()
        # allow overrides via kwargs / explicit args. metres_per_map_unit and
        # gate_threshold apply independently (both may be given together);
        # any other keyword must name a real EKFConfig field -- silently
        # ignoring a typo'd override would leave the filter running with the
        # wrong noise assumptions with no indication anything was wrong.
        if metres_per_map_unit is not None:
            cfg = replace(cfg, metres_per_map_unit=float(metres_per_map_unit))
        if gate_threshold is not None:
            cfg = replace(cfg, gate_threshold=float(gate_threshold))
        if kwargs:
            unknown = sorted(k for k in kwargs if k not in cfg.__dataclass_fields__)  # type: ignore[attr-defined]
            if unknown:
                raise TypeError(f"ESEKF() got unexpected EKFConfig override(s): {unknown}")
            cfg = replace(cfg, **kwargs)
        self.config: EKFConfig = cfg
        # nominal
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.ba = np.zeros(3, dtype=float)
        self.bg = np.zeros(3, dtype=float)
        self.P = np.eye(15, dtype=float)
        self.last_predict: float = 0.0
        self._initialized: bool = False
        self._age_frames: int = 0
        self._last_d2: float | None = None
        self._last_nu: np.ndarray | None = None
        self._last_R: np.ndarray | None = None
        self._last_source: str = "none"
        self._last_timestamp: float = 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_F_Qd(self, R: np.ndarray, dt: float, acc_m: np.ndarray | None = None, gyro_m: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        F = np.eye(15, dtype=float)
        # dp_dot = dv
        F[0:3, 3:6] = np.eye(3, dtype=float) * dt
        # dv_dot = -R * dba  (and -R*skew(acc)*dtheta if acc available)
        F[3:6, 9:12] = -R * dt
        if acc_m is not None:
            # dv wrt dtheta : -R * skew(acc_m - ba) * dt
            acc_corr = np.asarray(acc_m, dtype=float).reshape(3) - self.ba
            F[3:6, 6:9] = -R @ _skew(acc_corr) * dt
        # dtheta_dot = -R * dbg  (and -skew(gyro-bg)*dtheta if gyro available)
        F[6:9, 12:15] = -R * dt
        if gyro_m is not None:
            gyro_corr = np.asarray(gyro_m, dtype=float).reshape(3) - self.bg
            F[6:9, 6:9] = np.eye(3, dtype=float) - _skew(gyro_corr) * dt
        # biases random walk: identity ( already I15)
        Qd = np.zeros((15, 15), dtype=float)
        # scale by dt (continuous noise * dt)
        # clamp noise values to finite
        an = float(self.config.acc_noise) if math.isfinite(float(self.config.acc_noise)) else 0.1
        gn = float(self.config.gyro_noise) if math.isfinite(float(self.config.gyro_noise)) else 0.01
        barw = float(self.config.ba_rw) if math.isfinite(float(self.config.ba_rw)) else 1e-4
        bgrw = float(self.config.bg_rw) if math.isfinite(float(self.config.bg_rw)) else 1e-5
        Qd[3:6, 3:6] = np.eye(3, dtype=float) * (an * dt)
        Qd[6:9, 6:9] = np.eye(3, dtype=float) * (gn * dt)
        Qd[9:12, 9:12] = np.eye(3, dtype=float) * (barw * dt)
        Qd[12:15, 12:15] = np.eye(3, dtype=float) * (bgrw * dt)
        # small pos process to keep invertible
        # (optional) add tiny pos noise
        return F, Qd

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset_from_pose(
        self,
        p,
        q_wxyz=None,
        v=None,
        timestamp: float | None = None,
        **kwargs,
    ) -> None:
        """
        Init nominal and covariance.
        Accepts:
          reset_from_pose(p, q_wxyz, v, timestamp)
          reset_from_pose(p, q_wxyz, v=np.zeros(3), timestamp=0.0)
          reset_from_pose(p, q_wxyz, timestamp=0.0)  (v zero)
        Flexible for test call variations.
        """
        # handle overloaded positional where v may actually be timestamp, or q may be missing
        # kwargs aliases
        if q_wxyz is None:
            # maybe caller passed single dict?
            raise ValueError("reset_from_pose requires p and q_wxyz")
        # resolve timestamp aliases
        if timestamp is None:
            if "timestamp" in kwargs:
                timestamp = float(kwargs.pop("timestamp"))
            elif "stamp" in kwargs:
                timestamp = float(kwargs.pop("stamp"))
            elif "time" in kwargs:
                timestamp = float(kwargs.pop("time"))
            else:
                # if v is scalar numeric and timestamp not set, treat v as timestamp (2-arg form)
                if v is not None and isinstance(v, (float, int, np.floating)) and not isinstance(v, (list, tuple, np.ndarray)):
                    timestamp = float(v)
                    v = None
                else:
                    timestamp = 0.0
        else:
            try:
                timestamp = float(timestamp)
            except Exception:
                timestamp = 0.0

        # resolve velocity aliases if v still None
        if v is None:
            for k in ("velocity", "velocity_ned", "vel", "v_ned"):
                if k in kwargs:
                    v = kwargs.pop(k)
                    break
        if v is None:
            v_arr = np.zeros(3, dtype=float)
        else:
            try:
                v_arr = np.asarray(v, dtype=float).reshape(3)
                if not np.all(np.isfinite(v_arr)):
                    v_arr = np.zeros(3, dtype=float)
            except Exception:
                v_arr = np.zeros(3, dtype=float)

        try:
            p_arr = np.asarray(p, dtype=float).reshape(3)
            if not np.all(np.isfinite(p_arr)):
                p_arr = np.zeros(3, dtype=float)
        except Exception:
            p_arr = np.zeros(3, dtype=float)

        try:
            q_arr = _normalize_q(np.asarray(q_wxyz, dtype=float).reshape(4))
        except Exception:
            q_arr = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        # handle velocity scale if metres_per_map_unit set and caller gave NED m/s?
        # reset_from_pose v is assumed already map units/s; do not rescale here.
        # (predict will handle NED->map conversion for velocity_ned inputs)

        self.p = p_arr.astype(float)
        self.v = v_arr.astype(float)
        self.q = q_arr.astype(float)
        self.ba = np.zeros(3, dtype=float)
        self.bg = np.zeros(3, dtype=float)

        # init P = diag([pos_var, vel_var, att_var, ba_var, bg_var]) expanded
        pv = float(self.config.pos_var) if math.isfinite(float(self.config.pos_var)) else 0.5
        vv = float(self.config.vel_var) if math.isfinite(float(self.config.vel_var)) else 0.5
        av = float(self.config.att_var) if math.isfinite(float(self.config.att_var)) else 0.0076
        bav = float(self.config.ba_var) if math.isfinite(float(self.config.ba_var)) else 1e-2
        bgv = float(self.config.bg_var) if math.isfinite(float(self.config.bg_var)) else 1e-4
        diag = np.concatenate(
            [
                np.full(3, pv, dtype=float),
                np.full(3, vv, dtype=float),
                np.full(3, av, dtype=float),
                np.full(3, bav, dtype=float),
                np.full(3, bgv, dtype=float),
            ]
        )
        self.P = np.diag(diag)
        # enforce symmetry
        self.P = 0.5 * (self.P + self.P.T)

        self.last_predict = float(timestamp)
        self._last_timestamp = float(timestamp)
        self._initialized = True
        self._age_frames = 0
        self._last_d2 = None
        self._last_nu = None
        self._last_R = None
        self._last_source = "reset"

        # allow per-reset metres_per_map_unit override
        if "metres_per_map_unit" in kwargs:
            try:
                mpm = kwargs.pop("metres_per_map_unit")
                if mpm is not None:
                    self.config.metres_per_map_unit = float(mpm)  # type: ignore[assignment]
            except Exception:
                pass

    def predict(
        self,
        timestamp: float,
        velocity_ned=None,
        imu_sample=None,
        acc_m=None,
        gyro_m=None,
        metres_per_map_unit=None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Prediction step.

        Args:
          timestamp: target time (monotonic)
          velocity_ned: World NED m/s (3-tuple). If metres_per_map_unit is not None, converted to map units/s.
          imu_sample: alternative dict/tuple containing acc_m, gyro_m, velocity_ned
          acc_m, gyro_m: IMU measurements in body/camera frame
          metres_per_map_unit: override scale (float or None)

        Returns:
          (p_pred 3, q_pred 4, covariance 15x15 copy, dtheta_mag float)

        Behavior:
          dt = timestamp - last_predict, clamped to [0, max_dt] (default 0.5)
          p_pred = p + v*dt   (v is velocity_ned/m_map if velocity path, else current v)
          v_pred = velocity_ned (or hold)
          q_pred = q (no gyro in velocity mode); in IMU mode integrate gyro
          Propagate P via F=I+A*dt, Qd*dt
        """
        # flexible timestamp extraction if caller swapped args
        try:
            ts = float(timestamp)
        except Exception:
            ts = float(self.last_predict)

        # kwargs aliases for velocity
        if velocity_ned is None:
            for k in ("velocity", "vel_ned", "vel", "v_ned", "velocity_ned_world", "ned_velocity"):
                if k in kwargs:
                    velocity_ned = kwargs.pop(k)
                    break
        # imu_sample handling: may contain velocity_ned, acc, gyro
        if imu_sample is not None and velocity_ned is None and acc_m is None and gyro_m is None:
            # imu_sample could be dict with keys
            try:
                if isinstance(imu_sample, dict):
                    if "velocity_ned" in imu_sample:
                        velocity_ned = imu_sample.get("velocity_ned")
                    elif "vel_ned" in imu_sample:
                        velocity_ned = imu_sample.get("vel_ned")
                    if "acc" in imu_sample or "acc_m" in imu_sample:
                        acc_m = imu_sample.get("acc", imu_sample.get("acc_m"))
                    if "gyro" in imu_sample or "gyro_m" in imu_sample:
                        gyro_m = imu_sample.get("gyro", imu_sample.get("gyro_m"))
                    # also handle array-like imu_sample as velocity directly
                elif isinstance(imu_sample, (list, tuple, np.ndarray)):
                    arr = np.asarray(imu_sample, dtype=float)
                    if arr.size == 3:
                        # ambiguous: treat as velocity_ned
                        velocity_ned = arr
                    elif arr.size == 6:
                        acc_m = arr[0:3]
                        gyro_m = arr[3:6]
            except Exception:
                pass
        # also handle acc_m/gyro_m via kwargs aliases
        if acc_m is None:
            for k in ("acc", "accel", "acceleration"):
                if k in kwargs:
                    acc_m = kwargs.pop(k)
                    break
        if gyro_m is None:
            for k in ("gyro", "gyroscope", "angular_velocity"):
                if k in kwargs:
                    gyro_m = kwargs.pop(k)
                    break
        if metres_per_map_unit is None:
            # check kwargs or config
            if "metres_per_map_unit" in kwargs:
                metres_per_map_unit = kwargs.pop("metres_per_map_unit")
            elif "meters_per_map_unit" in kwargs:
                metres_per_map_unit = kwargs.pop("meters_per_map_unit")
            else:
                metres_per_map_unit = self.config.metres_per_map_unit
        else:
            try:
                metres_per_map_unit = float(metres_per_map_unit)
            except Exception:
                metres_per_map_unit = self.config.metres_per_map_unit

        if not self._initialized:
            # not yet init: return zeros but keep timestamp
            self.last_predict = ts
            return self.p.copy(), self.q.copy(), self.P.copy(), 0.0

        # compute dt clamped
        dt_raw = ts - float(self.last_predict)
        if not math.isfinite(dt_raw) or dt_raw <= 0:
            dt = 0.0
        else:
            dt = float(min(dt_raw, float(self.config.max_dt) if math.isfinite(float(self.config.max_dt)) else 0.5))

        # determine prediction mode
        has_velocity = velocity_ned is not None
        has_imu = acc_m is not None or gyro_m is not None

        # prepare velocity map units if needed
        v_map = None
        if has_velocity:
            try:
                vel_arr = np.asarray(velocity_ned, dtype=float).reshape(3)
                if not np.all(np.isfinite(vel_arr)):
                    has_velocity = False
                    v_map = None
                else:
                    if metres_per_map_unit is not None:
                        try:
                            mpm = float(metres_per_map_unit)
                            if math.isfinite(mpm) and abs(mpm) > 1e-9:
                                v_map = vel_arr / mpm
                            else:
                                v_map = vel_arr
                        except Exception:
                            v_map = vel_arr
                    else:
                        v_map = vel_arr
            except Exception:
                has_velocity = False
                v_map = None

        # normalize inputs for IMU
        acc_arr = None
        gyro_arr = None
        if has_imu:
            try:
                if acc_m is not None:
                    acc_arr = np.asarray(acc_m, dtype=float).reshape(3)
                    if not np.all(np.isfinite(acc_arr)):
                        acc_arr = None
            except Exception:
                acc_arr = None
            try:
                if gyro_m is not None:
                    gyro_arr = np.asarray(gyro_m, dtype=float).reshape(3)
                    if not np.all(np.isfinite(gyro_arr)):
                        gyro_arr = None
            except Exception:
                gyro_arr = None
            # if both missing, treat as no imu
            if acc_arr is None and gyro_arr is None:
                has_imu = False

        R = _quat_to_rot(self.q)

        dtheta_mag = 0.0

        if has_imu and (acc_arr is not None or gyro_arr is not None):
            # IMU prediction: p_dot=v, v_dot=R*(acc-ba)+g, q_dot=0.5*Omega(gyro-bg)*q
            # use simple Euler / mid-point (spec says mid-point dt)
            g_vec = np.asarray(self.config.g_world, dtype=float).reshape(3)
            if not np.all(np.isfinite(g_vec)):
                g_vec = np.zeros(3, dtype=float)
            # integrate orientation
            if gyro_arr is not None:
                gyro_corr = gyro_arr - self.bg
                dtheta = gyro_corr * dt
                dtheta_mag = float(np.linalg.norm(dtheta))
                dq = _exp_quat(dtheta)
                q_pred = _quat_mul(dq, self.q)
                q_pred = _normalize_q(q_pred)
            else:
                q_pred = self.q.copy()
                dtheta_mag = 0.0
            # integrate velocity and position
            if acc_arr is not None:
                acc_corr = acc_arr - self.ba
                # world accel = R*(acc_corr) + g ; R is world->cam so might need transpose but follow spec R*(acc-ba)+g
                # spec: v_dot = R(q)*(acc_m - ba) + g
                a_world = R @ acc_corr + g_vec
                # mid-point: use trapezoidal? spec says mid-point dt. Approx: v_pred = v + a*dt, p_pred = p + v*dt +0.5*a*dt^2
                v_pred = self.v + a_world * dt
                p_pred = self.p + self.v * dt + 0.5 * a_world * (dt * dt)
            else:
                v_pred = self.v.copy()
                p_pred = self.p + self.v * dt
            # propagate covariance
            F, Qd = self._build_F_Qd(R, dt, acc_arr, gyro_arr)
            # state update
            self.p = p_pred
            self.v = v_pred
            self.q = q_pred
            # P propagation
            try:
                self.P = F @ self.P @ F.T + Qd
                self.P = 0.5 * (self.P + self.P.T)
                # ensure positive: clamp diag non-negative
            except Exception:
                pass
        elif has_velocity and v_map is not None:
            # velocity mode: p_pred = p + v_map*dt, v_pred = v_map, q unchanged
            p_pred = self.p + v_map * dt
            v_pred = v_map.copy()
            q_pred = self.q.copy()
            dtheta_mag = 0.0
            F, Qd = self._build_F_Qd(R, dt, None, None)
            self.p = p_pred
            self.v = v_pred
            self.q = q_pred
            try:
                self.P = F @ self.P @ F.T + Qd
                self.P = 0.5 * (self.P + self.P.T)
            except Exception:
                pass
        else:
            # no external measurement: propagate with current velocity (constant velocity)
            p_pred = self.p + self.v * dt
            v_pred = self.v.copy()
            q_pred = self.q.copy()
            dtheta_mag = 0.0
            F, Qd = self._build_F_Qd(R, dt, None, None)
            self.p = p_pred
            # v unchanged
            self.q = q_pred
            try:
                self.P = F @ self.P @ F.T + Qd
                self.P = 0.5 * (self.P + self.P.T)
            except Exception:
                pass

        if math.isfinite(ts):
            # A non-finite ts already forced dt=0 above (no-op step); it must
            # not also overwrite last_predict, or one bad timestamp corrupts
            # dt_raw (and so every future step's dt) with a non-finite value
            # forever, silently freezing prediction from then on.
            self.last_predict = ts
            self._last_timestamp = ts
        self._age_frames += 1
        self._last_source = "predict_imu" if has_imu else ("predict_velocity" if has_velocity else "predict_hold")
        # quaternion normalize invariant
        self.q = _normalize_q(self.q)

        return self.p.copy(), self.q.copy(), self.P.copy(), float(dtheta_mag)

    def update_visual(
        self,
        z_p,
        z_q_wxyz,
        R_visual_pos=None,
        R_visual_ori=None,
        timestamp: float | None = None,
        **kwargs,
    ) -> tuple[bool, float, np.ndarray]:
        """
        Visual update: 6-dim measurement [p, q].

        Args:
          z_p: measured camera center (3,) map units
          z_q_wxyz: measured orientation wxyz (4,)
          R_visual_pos: pos covariance 3x3 or scalar variance
          R_visual_ori: ori covariance 3x3 or scalar (rad variance or deg std)
          timestamp: measurement time (optional, for info)

        Additional kwargs aliases:
          R_pos, R_ori, R_visual, R, inliers, reproj_rmse etc for adaptive path

        Returns:
          (accepted:bool, d2:float, nu:np.ndarray 6)
          On reject: nominal and P unchanged.
          On accept: nominal updated via Kalman, P Joseph form, error zeroed.
        """
        # handle aliases for covariance
        if R_visual_pos is None:
            for k in ("R_pos", "R_p", "Rpos", "pos_cov", "cov_pos", "R"):
                if k in kwargs:
                    R_visual_pos = kwargs.pop(k)
                    break
        if R_visual_ori is None:
            for k in ("R_ori", "R_q", "Rori", "ori_cov", "cov_ori", "yaw_cov", "R_yaw"):
                if k in kwargs:
                    R_visual_ori = kwargs.pop(k)
                    break
        # also allow full 6x6 R via kw "R"
        R_full_kw = None
        if "R_visual" in kwargs:
            R_full_kw = kwargs.pop("R_visual")
        if "R_6x6" in kwargs:
            R_full_kw = kwargs.pop("R_6x6")

        if timestamp is None:
            for k in ("stamp", "time", "t"):
                if k in kwargs:
                    try:
                        timestamp = float(kwargs.pop(k))
                    except Exception:
                        timestamp = None
                    break
        if timestamp is not None:
            try:
                timestamp = float(timestamp)
            except Exception:
                timestamp = None

        # gate threshold override
        gate_thr = float(self.config.gate_threshold)
        if "gate_threshold" in kwargs:
            try:
                gate_thr = float(kwargs.pop("gate_threshold"))
            except Exception:
                pass
        if "gate" in kwargs:
            try:
                gate_thr = float(kwargs.pop("gate"))
            except Exception:
                pass

        if not self._initialized:
            return False, float("inf"), np.zeros(6, dtype=float)

        # parse measurements
        try:
            zp = np.asarray(z_p, dtype=float).reshape(3)
        except Exception:
            return False, float("inf"), np.zeros(6, dtype=float)
        if not np.all(np.isfinite(zp)):
            return False, float("inf"), np.zeros(6, dtype=float)

        try:
            zq = _normalize_q(np.asarray(z_q_wxyz, dtype=float).reshape(4))
        except Exception:
            return False, float("inf"), np.zeros(6, dtype=float)

        # build R 6x6 block diag
        R = np.zeros((6, 6), dtype=float)
        # try full 6x6 first
        if R_full_kw is not None:
            try:
                Rf = np.asarray(R_full_kw, dtype=float)
                if Rf.shape == (6, 6):
                    R = Rf.copy()
                elif Rf.size == 6:
                    R = np.diag(Rf.reshape(6))
                else:
                    raise ValueError
            except Exception:
                R = np.zeros((6, 6), dtype=float)
        if np.all(R == 0):
            # build from pos/ori parts
            # pos part
            if R_visual_pos is None:
                # check adaptive params -> if inliers provided use adaptive
                # else fallback defaults
                has_adaptive_keys = any(k in kwargs for k in ("inliers", "reproj_rmse", "reproj_rms", "corr_count", "fb_error", "jump"))
                # also check if those passed as explicit kwargs before pop? use locals
                # For default, use base variances
                Rp = np.eye(3, dtype=float) * float(self.config.base_R_pos)
                Ro = np.eye(3, dtype=float) * ((math.radians(float(self.config.base_R_yaw_deg))) ** 2)
                # if adaptive info available via kwargs, try to scale (fallback)
                # we have inliers etc maybe still in kwargs
                # but R_visual_pos None and we have no adaptive info => use defaults
                if has_adaptive_keys:
                    inliers = kwargs.get("inliers", kwargs.get("num_inliers"))
                    reproj = kwargs.get("reproj_rmse", kwargs.get("reproj_rms", kwargs.get("reproj")))
                    corr = kwargs.get("corr_count", kwargs.get("corr"))
                    fb = kwargs.get("fb_error", kwargs.get("fb"))
                    jmp = kwargs.get("jump")
                    Rp_s, Ro_s = adaptive_visual_covariance(
                        inliers=inliers, reproj_rmse=reproj, corr_count=corr, fb_error=fb, jump=jmp,
                        base_R_pos=self.config.base_R_pos, base_R_yaw_deg=self.config.base_R_yaw_deg
                    )
                    Rp, Ro = Rp_s, Ro_s
                R[0:3, 0:3] = Rp
                R[3:6, 3:6] = Ro
            else:
                # parse pos
                try:
                    arr = np.asarray(R_visual_pos, dtype=float)
                    if arr.shape == (3, 3):
                        R[0:3, 0:3] = arr
                    elif arr.size == 3:
                        R[0:3, 0:3] = np.diag(arr.reshape(3))
                    elif arr.size == 1:
                        v = float(arr.ravel()[0])
                        # if variance seems large > 1 treat as variance, else variance; use directly
                        R[0:3, 0:3] = np.eye(3, dtype=float) * v
                    else:
                        R[0:3, 0:3] = np.eye(3, dtype=float) * float(self.config.base_R_pos)
                except Exception:
                    R[0:3, 0:3] = np.eye(3, dtype=float) * float(self.config.base_R_pos)

                # ori part
                if R_visual_ori is None:
                    Ro = np.eye(3, dtype=float) * ((math.radians(float(self.config.base_R_yaw_deg))) ** 2)
                    R[3:6, 3:6] = Ro
                else:
                    try:
                        arr = np.asarray(R_visual_ori, dtype=float)
                        if arr.shape == (3, 3):
                            R[3:6, 3:6] = arr
                        elif arr.size == 3:
                            R[3:6, 3:6] = np.diag(arr.reshape(3))
                        elif arr.size == 1:
                            v = float(arr.ravel()[0])
                            # heuristic: if v > 1, likely degrees std, convert to rad variance; if small (<0.5) likely rad variance already
                            if v > 0.5 and v < 180:  # degrees std
                                var = (math.radians(v)) ** 2
                                R[3:6, 3:6] = np.eye(3, dtype=float) * var
                            else:
                                R[3:6, 3:6] = np.eye(3, dtype=float) * v
                        else:
                            R[3:6, 3:6] = np.eye(3, dtype=float) * ((math.radians(float(self.config.base_R_yaw_deg))) ** 2)
                    except Exception:
                        R[3:6, 3:6] = np.eye(3, dtype=float) * ((math.radians(float(self.config.base_R_yaw_deg))) ** 2)

            # if R still zero (fallback)
            if np.trace(R) == 0:
                R[0:3, 0:3] = np.eye(3) * 0.3
                R[3:6, 3:6] = np.eye(3) * ((math.radians(5.0)) ** 2)

        # ensure R finite and positive
        R = 0.5 * (R + R.T)
        # add tiny epsilon to diag for invertibility
        eps = 1e-9
        for i in range(6):
            if not math.isfinite(R[i, i]) or R[i, i] <= 0:
                R[i, i] = 1e-6 if i < 3 else (math.radians(5.0) ** 2)
            # ensure minimal
            if R[i, i] < eps:
                R[i, i] = eps

        # innovations
        nu_p = zp - self.p
        q_pred_inv = _quat_inv(self.q)
        q_err = _quat_mul(q_pred_inv, zq)
        nu_q = _log_quat(q_err)  # 3-vector
        nu = np.concatenate([nu_p, nu_q]).astype(float)  # 6

        # H 6x15
        H = np.zeros((6, 15), dtype=float)
        H[0:3, 0:3] = np.eye(3, dtype=float)
        H[3:6, 6:9] = np.eye(3, dtype=float)

        # S = H P H^T + R  (6x6)
        try:
            S = H @ self.P @ H.T + R
            S = 0.5 * (S + S.T)
            # invert 6x6
            invS = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # fallback pseudo
            try:
                invS = np.linalg.pinv(S)
            except Exception:
                return False, float("inf"), nu
        except Exception:
            return False, float("inf"), nu

        # mahalanobis
        try:
            d2 = float(nu.T @ invS @ nu)
        except Exception:
            d2 = float("inf")
        if not math.isfinite(d2):
            d2 = float("inf")

        # gating
        if d2 > gate_thr:
            # reject, store but not update
            self._last_d2 = d2
            self._last_nu = nu.copy()
            self._last_R = R.copy()
            self._last_source = "reject"
            if timestamp is not None and math.isfinite(timestamp):
                self._last_timestamp = float(timestamp)
            return False, d2, nu

        # accept: K = P H^T invS  (15x6)
        try:
            PHt = self.P @ H.T  # 15x6
            K = PHt @ invS  # 15x6
        except Exception:
            return False, d2, nu

        delta = K @ nu  # 15

        # update nominal
        dp = delta[0:3]
        dv = delta[3:6]
        dtheta = delta[6:9]
        dba = delta[9:12]
        dbg = delta[12:15]

        self.p = self.p + dp
        self.v = self.v + dv
        # q = Exp(dtheta) * q
        if float(np.linalg.norm(dtheta)) > 1e-12:
            dq = _exp_quat(dtheta)
            self.q = _quat_mul(dq, self.q)
        self.q = _normalize_q(self.q)
        self.ba = self.ba + dba
        self.bg = self.bg + dbg

        # Joseph form: P = (I - K H) P (I - K H)^T + K R K^T
        try:
            I15 = np.eye(15, dtype=float)
            KH = K @ H  # 15x15
            I_KH = I15 - KH
            KRK = K @ R @ K.T
            self.P = I_KH @ self.P @ I_KH.T + KRK
            self.P = 0.5 * (self.P + self.P.T)
            # ensure PSD: clamp tiny negative diag from numeric
            # (optional) force diag positive
            for i in range(15):
                if self.P[i, i] < 1e-12:
                    self.P[i, i] = 1e-12
        except Exception:
            pass

        self._age_frames = 0
        self._last_d2 = d2
        self._last_nu = nu.copy()
        self._last_R = R.copy()
        self._last_source = "visual"
        if timestamp is not None and math.isfinite(timestamp):
            self._last_timestamp = float(timestamp)
            # also update last_predict to timestamp if newer? but keep separate
            # visual update does not move last_predict forward automatically unless timestamp > last_predict
            # keep as is; do not overwrite prediction clock

        # error state reset implicit (delta applied)
        return True, d2, nu

    def get_state(self) -> dict:
        return {
            "p": self.p.copy(),
            "v": self.v.copy(),
            "q": self.q.copy(),
            "ba": self.ba.copy(),
            "bg": self.bg.copy(),
            "timestamp": float(self.last_predict),
            # aliases for convenience
            "position": self.p.copy(),
            "velocity": self.v.copy(),
            "orientation": self.q.copy(),
        }

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def prediction_allowed(self) -> bool:
        if not self._initialized:
            return False
        if self._age_frames >= int(self.config.max_age_frames):
            return False
        # covariance trace
        try:
            pos_trace = float(np.trace(self.P[0:3, 0:3]))
            if not math.isfinite(pos_trace) or pos_trace > float(self.config.pos_trace_threshold):
                return False
        except Exception:
            return False
        # yaw sigma < threshold
        try:
            yaw_thr_rad = math.radians(float(self.config.yaw_sigma_threshold_deg))
            # extract yaw variance approx from dtheta covariance
            # use trace/3 or element (2,2) for yaw axis
            dtheta_cov = self.P[6:9, 6:9]
            # average variance
            yaw_var = float(np.trace(dtheta_cov) / 3.0)
            # alternative: use z variance if smaller?
            # keep max of both to be conservative
            z_var = float(dtheta_cov[2, 2]) if dtheta_cov.shape == (3, 3) else yaw_var
            var = max(yaw_var, z_var) if math.isfinite(z_var) else yaw_var
            yaw_sigma = math.sqrt(max(var, 0.0))
            if not math.isfinite(yaw_sigma) or yaw_sigma > yaw_thr_rad:
                return False
        except Exception:
            return False
        # state valid: finite
        if not np.all(np.isfinite(self.p)) or not np.all(np.isfinite(self.q)):
            return False
        if not np.all(np.isfinite(self.P)):
            return False
        return True

    def as_info(self) -> dict:
        # P diag
        try:
            p_diag = np.diag(self.P).copy()
        except Exception:
            p_diag = np.zeros(15, dtype=float)
        info = {
            "timestamp": float(self._last_timestamp if self._last_timestamp is not None else self.last_predict),
            "p": self.p.copy().tolist(),
            "q": self.q.copy().tolist(),
            "v": self.v.copy().tolist(),
            "ba": self.ba.copy().tolist(),
            "bg": self.bg.copy().tolist(),
            "P_diag": p_diag.tolist() if hasattr(p_diag, "tolist") else list(p_diag),
            "P_trace_pos": float(np.trace(self.P[0:3, 0:3])) if self.P.shape == (15, 15) else None,
            "d2": float(self._last_d2) if self._last_d2 is not None and math.isfinite(float(self._last_d2)) else None,
            "innovation": self._last_nu.copy().tolist() if self._last_nu is not None else None,
            "nu": self._last_nu.copy().tolist() if self._last_nu is not None else None,
            "source": str(self._last_source),
            "age_frames": int(self._age_frames),
            "prediction_allowed": bool(self.prediction_allowed()),
            "last_predict": float(self.last_predict),
        }
        return info

    # method alias for adaptive (instance)
    def adaptive_visual_covariance(
        self,
        inliers=None,
        reproj_rmse=None,
        corr_count=None,
        fb_error=None,
        jump=None,
        base_R_pos: float | np.ndarray = 0.3,
        base_R_yaw: float | None = None,
        base_R_yaw_deg: float | None = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        # use instance defaults if not provided
        if base_R_pos == 0.3 and self.config.base_R_pos != 0.3:
            base_R_pos = self.config.base_R_pos  # type: ignore[assignment]
        if base_R_yaw_deg is None and base_R_yaw is None:
            base_R_yaw_deg = self.config.base_R_yaw_deg
        return adaptive_visual_covariance(
            inliers=inliers,
            reproj_rmse=reproj_rmse,
            corr_count=corr_count,
            fb_error=fb_error,
            jump=jump,
            base_R_pos=base_R_pos,
            base_R_yaw=base_R_yaw,
            base_R_yaw_deg=base_R_yaw_deg,
            **kwargs,
        )

    # convenience: direct accessors for tests
    @property
    def state(self):
        return self.get_state()

    @property
    def covariance(self):
        return self.get_covariance()


__all__ = ["ESEKF", "EKFConfig", "adaptive_visual_covariance"]

