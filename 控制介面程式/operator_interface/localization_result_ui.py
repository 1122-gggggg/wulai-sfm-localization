"""UI-facing localization result adapters for the desktop operator interface."""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from localization_contract import InvalidLocalizationResult, LocalizationResult


def _stabilized_yaw(
    raw_pose: dict, yaw_update: object, stamp: float
) -> tuple[str | None, float | None, object | None]:
    yaw_key = (
        "yaw_raw" if "yaw_raw" in raw_pose
        else "yaw" if "yaw" in raw_pose
        else None
    )
    if not callable(yaw_update) or yaw_key is None:
        return yaw_key, None, None
    try:
        raw_yaw = float(raw_pose[yaw_key])
    except (TypeError, ValueError, OverflowError):
        return yaw_key, None, None
    if not math.isfinite(raw_yaw):
        return yaw_key, None, None
    filtered_yaw, info = yaw_update(raw_yaw, stamp)
    return yaw_key, float(filtered_yaw), info


def stabilize_live_result_pose(
    owner: Any, result: dict, xyz: np.ndarray | None
) -> np.ndarray | None:
    """Filter the UI/control pose while retaining the raw worker pose."""
    if xyz is None:
        return xyz
    pose_value = result.get("pose")
    if not isinstance(pose_value, dict):
        return xyz
    xyz_stabilizer = getattr(owner, "pose_stabilizer", None)
    yaw_stabilizer = getattr(owner, "yaw_stabilizer", None)
    yaw_update = getattr(yaw_stabilizer, "update", None)
    if xyz_stabilizer is None and not callable(yaw_update):
        return xyz
    raw_pose = dict(pose_value)
    stamp = localization_pose_timestamp(result)
    if stamp is None:
        stamp = time.monotonic()
    filtered_xyz = xyz
    filter_info = None
    if xyz_stabilizer is not None:
        filtered_xyz, filter_info = xyz_stabilizer.update(xyz, float(stamp))
    filtered_pose = dict(raw_pose)
    if xyz_stabilizer is not None:
        filtered_pose.update({
            "x": float(filtered_xyz[0]),
            "y": float(filtered_xyz[1]),
            "z": float(filtered_xyz[2]),
        })
    yaw_key, filtered_yaw, yaw_filter_info = _stabilized_yaw(
        raw_pose, yaw_update, float(stamp)
    )
    if yaw_key is not None and filtered_yaw is not None:
        # The original stays in pose_raw; public pose keeps the legacy key.
        filtered_pose[yaw_key] = filtered_yaw
    result["pose_raw"] = raw_pose
    result["pose"] = filtered_pose
    if filter_info is not None:
        result["pose_filter"] = filter_info
    if yaw_filter_info is not None:
        result["yaw_filter"] = yaw_filter_info
    return filtered_xyz


def normalize_live_localization_result(result: object) -> tuple[dict, np.ndarray | None]:
    """Downgrade a worker success unless it contains a complete finite XYZ pose."""
    if not isinstance(result, dict):
        return {"success": False, "error": "invalid localization result: expected object"}, None
    normalized = dict(result)
    pose = normalized.get("pose")
    xyz = None
    try:
        if isinstance(pose, dict):
            xyz = np.asarray([float(pose[name]) for name in ("x", "y", "z")], dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                xyz = None
    except (KeyError, TypeError, ValueError, OverflowError):
        xyz = None
    if not normalized.get("success"):
        return normalized, xyz
    if xyz is None:
        normalized["success"] = False
        normalized["error"] = "invalid localization pose: x/y/z must be finite"
        return normalized, None
    return normalized, xyz


def localization_result_display_seq(result: dict) -> int | None:
    """Return the client publication order, with a legacy worker fallback."""
    display_seq = result.get("display_seq")
    if display_seq is None:
        display_seq = result.get("seq")
    if isinstance(display_seq, bool) or not isinstance(display_seq, int):
        return None
    return display_seq if display_seq >= 0 else None


def validate_live_localization_result(result: object) -> dict:
    """Validate new worker payloads once while retaining dict compatibility."""
    if not isinstance(result, dict):
        return {
            "success": False,
            "validity": False,
            "mode": "LOST",
            "next_mode": "LOST",
            "localization_exception": True,
            "error": "invalid localization result: expected object",
        }
    if result.get("localization_contract_version") != 1:
        # Older test doubles/plugins remain readable until they opt into the
        # explicit boundary fields. Production worker payloads always carry v1.
        return dict(result)
    try:
        return LocalizationResult.from_payload(result).to_payload()
    except InvalidLocalizationResult as exc:
        invalid = dict(result)
        invalid.update({
            "success": False,
            "validity": False,
            "mode": "LOST",
            "next_mode": "LOST",
            "localization_exception": True,
            "error": f"invalid localization result: {exc}",
        })
        return invalid


class TemporalPoseStabilizer:
    """Causal robust filter for the pose exposed to UI/control consumers.

    Raw PnP stays inside the tracker for candidate selection and is retained in
    telemetry.  A 3-sample median removes one-frame flips; a short time-based
    low-pass and rate limit prevent the remaining measurement noise from being
    published as impossible vehicle motion.
    """

    def __init__(self, tau_s: float = 0.15, max_speed_u_s: float = 2.0,
                 step_slack_u: float = 0.03, max_step_u: float = 0.15):
        self.tau_s = max(1e-3, float(tau_s))
        self.max_speed_u_s = max(0.0, float(max_speed_u_s))
        self.step_slack_u = max(0.0, float(step_slack_u))
        self.max_step_u = max(0.0, float(max_step_u))
        self._raw_history: list[np.ndarray] = []
        self._filtered: np.ndarray | None = None
        self._stamp: float | None = None

    def reset(self) -> None:
        self._raw_history.clear()
        self._filtered = None
        self._stamp = None

    def update(self, xyz: np.ndarray, stamp: float) -> tuple[np.ndarray, dict]:
        raw = np.asarray(xyz, dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError("pose stabilizer requires finite XYZ")
        now = float(stamp)
        if not math.isfinite(now):
            raise ValueError("pose stabilizer requires a finite timestamp")

        self._raw_history.append(raw.copy())
        del self._raw_history[:-3]
        median = np.median(np.asarray(self._raw_history), axis=0)
        if self._filtered is None:
            self._filtered = median.copy()
            self._stamp = now
            return self._filtered.copy(), {
                "enabled": True, "raw_delta_u": 0.0,
                "published_delta_u": 0.0, "limited": False,
            }

        elapsed = now - float(self._stamp)
        dt = min(0.15, max(0.001, elapsed if math.isfinite(elapsed) else 0.001))
        alpha = 1.0 - math.exp(-dt / self.tau_s)
        target = self._filtered + alpha * (median - self._filtered)
        delta = target - self._filtered
        raw_delta = float(np.linalg.norm(raw - self._filtered))
        distance = float(np.linalg.norm(delta))
        limit = self.step_slack_u + self.max_speed_u_s * dt
        if self.max_step_u > 0:
            limit = min(limit, self.max_step_u)
        limited = bool(limit > 0 and distance > limit)
        if limited:
            delta *= limit / max(distance, 1e-12)
        self._filtered = self._filtered + delta
        self._stamp = now
        return self._filtered.copy(), {
            "enabled": True,
            "dt_s": dt,
            "alpha": alpha,
            "raw_delta_u": raw_delta,
            "published_delta_u": float(np.linalg.norm(delta)),
            "step_limit_u": limit,
            "limited": limited,
        }


def _wrap_yaw(angle: float) -> float:
    """Wrap an angle to the conventional ``[-pi, pi]`` interval."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    # Keep a positive input at +pi instead of changing its representation to
    # -pi.  The two values are equivalent, but retaining the input side makes
    # the first published sample unsurprising.
    if wrapped == -math.pi and angle > 0.0:
        return math.pi
    return wrapped


def _yaw_delta(angle: float, reference: float) -> float:
    """Return the shortest signed circular difference between two angles."""
    return (float(angle) - float(reference) + math.pi) % (2.0 * math.pi) - math.pi


class TemporalYawStabilizer:
    """Causal, circular, one-frame-robust filter for ``pose.yaw_raw``.

    The trailing odd-sized median is taken in the local angular neighbourhood
    of the previous filtered value, so samples on opposite sides of +/-pi stay
    adjacent.  A short circular EMA then smooths ordinary measurement noise;
    no tracker or EDM work is added.
    """

    def __init__(self, tau_s: float = 0.15, window_size: int = 3):
        self.tau_s = max(1e-3, float(tau_s))
        size = max(3, int(window_size))
        self.window_size = size if size % 2 else size + 1
        self._raw_history: list[float] = []
        self._filtered: float | None = None
        self._stamp: float | None = None

    def reset(self) -> None:
        self._raw_history.clear()
        self._filtered = None
        self._stamp = None

    def update(
            self, yaw_raw: float, stamp: float | None = None,
    ) -> tuple[float, dict]:
        raw = float(yaw_raw)
        if not math.isfinite(raw):
            raise ValueError("yaw stabilizer requires a finite angle")
        now = time.monotonic() if stamp is None else float(stamp)
        if not math.isfinite(now):
            raise ValueError("yaw stabilizer requires a finite timestamp")

        raw = _wrap_yaw(raw)
        self._raw_history.append(raw)
        del self._raw_history[:-self.window_size]
        reference = raw if self._filtered is None else self._filtered
        offsets = [_yaw_delta(value, reference) for value in self._raw_history]
        # Padding an incomplete window with the previous output means a second
        # sample cannot move the filter on its own.  This is the causal
        # one-frame spike guard; once a third sample arrives, the median is
        # robust to either side of a single outlier.
        offsets.extend([0.0] * (self.window_size - len(offsets)))
        target_delta = float(np.median(np.asarray(offsets, dtype=float)))
        target = _wrap_yaw(reference + target_delta)
        raw_delta = abs(_yaw_delta(raw, reference))
        previous = self._filtered
        if previous is None:
            filtered = target
            dt = None
            alpha = 1.0
            published_delta = 0.0
        else:
            elapsed = now - float(self._stamp)
            dt = min(0.15, max(0.001, elapsed if math.isfinite(elapsed) else 0.001))
            alpha = 1.0 - math.exp(-dt / self.tau_s)
            filtered = _wrap_yaw(
                previous + alpha * _yaw_delta(target, previous)
            )
            published_delta = abs(_yaw_delta(filtered, previous))
        self._filtered = filtered
        self._stamp = now
        return filtered, {
            "enabled": True,
            "dt_s": dt,
            "alpha": alpha,
            "raw_delta_rad": raw_delta,
            "published_delta_rad": published_delta,
            "target_delta_rad": abs(target_delta),
            "spike_rejected": bool(
                len(self._raw_history) == self.window_size
                and raw_delta - abs(target_delta) > 0.5
            ),
        }


def annotate_ui_arrival_timing(result: dict, now: float | None = None) -> dict:
    """Attach UI-consumption timing without calling a stream stamp capture time."""
    arrival_ns = time.monotonic_ns() if now is None else int(round(float(now) * 1e9))
    arrival = arrival_ns * 1e-9
    result["ui_arrival_mono"] = arrival
    result["ui_arrival_mono_ns"] = arrival_ns

    def elapsed_ms(start_key: str, out_key: str) -> None:
        start = result.get(start_key)
        if start is None:
            return
        try:
            result[out_key] = max(0.0, (arrival - float(start)) * 1000.0)
        except (TypeError, ValueError, OverflowError):
            pass

    elapsed_ms("client_submit_mono", "e2e_submit_to_ui_ms")
    elapsed_ms("client_response_mono", "ui_poll_delay_ms")
    # This stamp may be NTP-mapped or callback-receipt time. Keep the neutral
    # "source stamp" name; it is not guaranteed camera capture latency.
    elapsed_ms("source_frame_stamp_mono", "source_stamp_age_at_ui_ms")
    callback_ns = result.get("frame_callback_enter_mono_ns")
    if callback_ns is not None:
        try:
            result["callback_to_ui_ms"] = max(
                0.0, (arrival_ns - int(callback_ns)) / 1_000_000.0)
        except (TypeError, ValueError, OverflowError):
            pass
    return result


def localization_pose_timestamp(
    result: dict,
    *,
    arrival_mono: float | None = None,
) -> float | None:
    """Return the conservative monotonic timestamp for a published pose.

    A worker response can arrive well after the frame it localized.  Source and
    capture timestamps therefore take precedence over UI arrival; the earliest
    valid timing marker is used so either capture age or processing/queue delay
    makes the pose stale.  Older payloads without timing fields fall back to
    their UI arrival timestamp for compatibility.
    """
    candidates: list[float] = []

    def add(value: object, *, nanoseconds: bool = False) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if nanoseconds:
            number *= 1e-9
        if math.isfinite(number) and number > 0.0:
            candidates.append(number)

    for key in (
        "source_frame_stamp_mono",
        "worker_core_done_mono",
        "client_submit_mono",
        "client_response_mono",
    ):
        add(result.get(key))
    for key in (
        "source_frame_stamp_mono_ns",
        "capture_mono_ns",
        "pose_mono_ns",
        "worker_core_done_mono_ns",
        "client_submit_mono_ns",
        "client_response_mono_ns",
    ):
        add(result.get(key), nanoseconds=True)
    if candidates:
        return min(candidates)

    fallback = result.get("ui_arrival_mono", arrival_mono)
    try:
        value = float(fallback)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def localization_result_is_weak(result: dict) -> bool:
    state = str(result.get("next_mode") or result.get("mode") or "").upper()
    return bool(result.get("weak")) or state in {
        "WEAK", "WEAK_TRACK", "LOST", "FAIL", "TRACK_FAIL", "TRACK_LOST",
    }
