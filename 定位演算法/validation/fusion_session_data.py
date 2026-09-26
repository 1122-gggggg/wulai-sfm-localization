"""Read recorded camera poses and independent telemetry clocks for offline fusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"expected JSON object in {path}")
                yield row


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def camera_pose(row):
    if (
        row.get("success") is not True
        or row.get("direct_status") not in {"FAST_TRACK", "RELOC_SEED"}
        or row.get("confidence_low")
        or row.get("imu_bridge")
        or row.get("reseed_confirming")
    ):
        return None
    pose = row.get("pose_raw") or row.get("pose")
    try:
        center = np.array([pose[key] for key in ("x", "y", "z")], dtype=float)
        # Logged right/down/forward world axes are the ROWS of R_cam_from_world.
        rotation = np.asarray(row["camera_axes_world"], dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(center).all()
            or not np.isfinite(rotation).all()
            or np.linalg.det(rotation) <= 0
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4)
        ):
            return None
        return np.column_stack((rotation, -rotation @ center))
    except (KeyError, TypeError, ValueError):
        return None


def _velocity_sample(row):
    try:
        stamp = float(row["speed_mono_ns"]) * 1e-9
        receipt = float(row["t_mono_ns"]) * 1e-9
        vector = [
            float(row[key]) for key in ("speed_north_mps", "speed_east_mps", "speed_down_mps")
        ]
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([stamp, receipt, *vector]).all() or not 0 <= receipt - stamp <= 0.5:
        return None
    return (stamp, *vector)


def _read_frames(visual_path):
    frames, bases, identities = {}, [], set()
    for row in read_rows(visual_path):
        if row.get("event") == "auto_route_plan":
            frame = row["control_config"]["map_frame"]
            bases.append(np.array([frame[key] for key in ("east", "north", "up")]))
            identities.add(row.get("coordinate_frame_id"))
        if row.get("event") != "pose_result":
            continue
        stamp = row.get("source_frame_stamp_mono")
        if not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
            continue
        frames.setdefault(
            stamp, {"stamp": stamp, "pose": camera_pose(row), "status": row.get("direct_status")}
        )
    return frames, bases, identities


def _read_velocities(telemetry_path):
    velocities = {}
    for row in read_rows(telemetry_path):
        if row.get("event") == "control_motion_history":
            for north, east, stamp in row.get("horizontal_ned_samples", []):
                # Shared replay consumers fuse horizontal velocity only. The
                # unused down slot is zero, not an observed vertical speed.
                sample = _velocity_sample(
                    {
                        "speed_mono_ns": stamp * 1e9,
                        "t_mono_ns": row["t_mono_ns"],
                        "speed_north_mps": north,
                        "speed_east_mps": east,
                        "speed_down_mps": 0.0,
                    }
                )
                if sample is not None:
                    velocities.setdefault(sample[0], sample)
        if row.get("event") == "fused_odometry":
            sample = _velocity_sample(row)
            if sample is not None:
                velocities.setdefault(sample[0], sample)
    return np.array(sorted(velocities.values()), dtype=float).reshape(-1, 4)


def load_session(session):
    session = Path(session)
    visual_path = session / "localization.jsonl"
    telemetry_path = session / "telemetry.jsonl"
    if not visual_path.is_file():
        visual_path = session / "trajectory.jsonl"
    if not telemetry_path.is_file():
        telemetry_path = session / "trajectory.jsonl"
    frames, bases, identities = _read_frames(visual_path)
    velocities = _read_velocities(telemetry_path)
    if not bases or len(identities) != 1:
        raise ValueError("one recorded map frame and route plan are required")
    basis = bases[0]
    if (
        basis.shape != (3, 3)
        or not np.isfinite(basis).all()
        or not np.allclose(basis @ basis.T, np.eye(3), atol=1e-5)
        or np.linalg.det(basis) < 0.999
        or any(not np.allclose(basis, other) for other in bases[1:])
    ):
        raise ValueError("recorded map axes must be consistent and orthonormal")
    identity = {str(path): file_digest(path) for path in {visual_path, telemetry_path}}
    return (
        sorted(frames.values(), key=lambda r: r["stamp"]),
        velocities,
        basis,
        identity,
    )


def integrated_velocity(velocities, start, end):
    """Integrate NED endpoints, refusing missing coverage and sensor gaps."""
    if len(velocities) < 2 or start < velocities[0, 0] or end > velocities[-1, 0]:
        return None
    t = velocities[:, 0]
    overlaps = (t[:-1] < end) & (t[1:] > start)
    if np.any(np.diff(t)[overlaps] > 0.5):
        return None
    knots = np.concatenate(([start], t[(t > start) & (t < end)], [end]))
    values = np.column_stack([np.interp(knots, t, velocities[:, i]) for i in (1, 2, 3)])
    return np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(knots)[:, None], axis=0)


def calibration_segments(frames, velocities, basis, cutoff):
    """Non-overlapping motion intervals from the calibration prefix only."""
    observed = [row for row in frames if row["pose"] is not None and row["stamp"] <= cutoff]
    segments, anchor, previous = [], None, None
    for row in observed:
        pose, stamp = row["pose"], row["stamp"]
        center = -pose[:, :3].T @ pose[:, 3]
        if previous is None or stamp - previous > 0.5:
            anchor = (stamp, center)
        previous = stamp
        if stamp - anchor[0] < 2.0:
            continue
        displacement = integrated_velocity(velocities, anchor[0], stamp)
        visual = basis[:2] @ (center - anchor[1])
        anchor = (stamp, center)
        if displacement is None or np.linalg.norm(displacement[:2]) < 0.5:
            continue
        if np.linalg.norm(visual) >= 0.02:
            segments.append((visual, displacement[[1, 0]]))  # metric EN
    return segments


def fit_horizontal_calibration(segments):
    """Estimate scale/yaw from telemetry, never label it independent calibration."""
    if len(segments) < 8:
        return {"ready": False, "reason": "fewer than eight independent motion intervals"}
    visual, metric = np.asarray([s[0] for s in segments]), np.asarray([s[1] for s in segments])
    distance = float(np.linalg.norm(metric, axis=1).sum())
    if distance < 10.0:
        return {"ready": False, "reason": "less than ten metres of telemetry displacement"}
    ratio = np.linalg.norm(metric, axis=1) / np.linalg.norm(visual, axis=1)
    scale = float(np.median(ratio))
    angles = np.arctan2(visual[:, 1], visual[:, 0]) - np.arctan2(metric[:, 1], metric[:, 0])
    theta = float(np.arctan2(np.sin(angles).sum(), np.cos(angles).sum()))
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    residuals = np.linalg.norm(visual * scale - metric @ rotation.T, axis=1)
    spread = float((np.percentile(ratio, 75) - np.percentile(ratio, 25)) / scale)
    angle_error = np.abs(np.angle(np.exp(1j * (angles - theta))))
    half = len(ratio) // 2
    drift = float(np.median(ratio[half:]) / np.median(ratio[:half]))
    ready = bool(
        spread <= 0.2
        and abs(drift - 1.0) <= 0.2
        and np.percentile(angle_error, 75) <= np.deg2rad(20.0)
    )
    return {
        "ready": ready,
        "reason": "consistent prefix" if ready else "inconsistent scale/yaw",
        "scale_m_per_unit": scale,
        "rotation_en_to_map": rotation.tolist(),
        "yaw_offset_rad": theta,
        "segments": len(segments),
        "distance_m": distance,
        "relative_iqr": spread,
        "late_early_scale_ratio": drift,
        "residual_p95_m": float(np.percentile(residuals, 95)),
        "independent_calibration": False,
        "vertical_calibration_verified": False,
    }
