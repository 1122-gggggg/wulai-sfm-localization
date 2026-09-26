#!/usr/bin/env python3
"""Read-only first look at an IMU flight-test session.

Checks the completeness and timing of fused telemetry and recorded images.
The current DIRECT backend has a bounded fused-yaw bridge, not ESEKF or raw
inertial integration. A usable recording is not an accuracy or flight approval.

Reads only ``telemetry.jsonl``, ``localization.jsonl`` and ``imu_test/`` from
the session directory. Writes nothing unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterator

_FLIGHT_CONTROL = Path(__file__).resolve().parents[1] / "定位演算法" / "flight_control"
if str(_FLIGHT_CONTROL) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_CONTROL))
try:
    from map_scale import estimate_map_scale
except ImportError:  # estimator unavailable: scale section reports as such
    estimate_map_scale = None  # type: ignore[assignment]

#: Same bound live_localizer_protocol enforces before a fused sample is allowed
#: onto the wire. A frame whose telemetry is older than this never reached the
#: tracker, however good the rest of the recording looks.
MAX_FUSED_SYNC_ERROR_S = 0.15
#: Below this the aircraft was parked. Attitude that never leaves a degree and
#: velocity that never leaves a few cm/s carry no information for the filter.
MIN_ATTITUDE_SPAN_DEG = 5.0
MIN_SPEED_SPAN_MPS = 0.5
#: One usable episode of each is the floor for a recovery comparison.
MIN_LOST_EPISODES = 1


def read_jsonl(path: Path, errors: list[str] | None = None) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                if errors is not None:
                    errors.append(f"{path.name}:{number}: invalid JSON")
                continue
            if isinstance(record, dict):
                yield record


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _span(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _rate_hz(stamps: list[float]) -> float | None:
    if len(stamps) < 2:
        return None
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 0 else None


def _largest_gap_s(stamps: list[float]) -> float | None:
    if len(stamps) < 2:
        return None
    return max(b - a for a, b in zip(stamps, stamps[1:]))


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------
def summarize_imu(records: list[dict[str, Any]]) -> dict[str, Any]:
    stamps: list[float] = []
    roll: list[float] = []
    pitch: list[float] = []
    yaw: list[float] = []
    speed: list[float] = []
    vertical: list[float] = []
    with_attitude = 0
    with_velocity = 0
    for record in records:
        stamp = _finite(record.get("t_mono_ns"))
        if stamp is not None:
            stamps.append(stamp * 1e-9)
        attitude = [_finite(record.get(key)) for key in ("att_roll", "att_pitch", "att_yaw")]
        if all(value is not None for value in attitude):
            with_attitude += 1
            roll.append(math.degrees(attitude[0]))  # type: ignore[arg-type]
            pitch.append(math.degrees(attitude[1]))  # type: ignore[arg-type]
            yaw.append(math.degrees(attitude[2]))  # type: ignore[arg-type]
        velocity = [
            _finite(record.get(key))
            for key in ("speed_north_mps", "speed_east_mps", "speed_down_mps")
        ]
        if all(value is not None for value in velocity):
            with_velocity += 1
            speed.append(math.hypot(velocity[0], velocity[1]))  # type: ignore[arg-type]
            vertical.append(abs(velocity[2]))  # type: ignore[arg-type]
    stamps.sort()
    return {
        "samples": len(records),
        "with_attitude": with_attitude,
        "with_velocity": with_velocity,
        "rate_hz": _rate_hz(stamps),
        "duration_s": (stamps[-1] - stamps[0]) if len(stamps) >= 2 else 0.0,
        "largest_gap_s": _largest_gap_s(stamps),
        "roll_span_deg": _span(roll),
        "pitch_span_deg": _span(pitch),
        # Yaw wraps at +-pi, so a span across the wrap reads as ~360. That
        # overstates rather than hides motion, which is the safe direction for
        # a "did the aircraft actually turn" check.
        "yaw_span_deg": _span(yaw),
        "ground_speed_max_mps": max(speed) if speed else 0.0,
        "vertical_speed_max_mps": max(vertical) if vertical else 0.0,
    }


def summarize_sticks(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_axis: dict[str, list[int]] = {}
    moved = 0
    active = 0
    for record in records:
        if record.get("moved"):
            moved += 1
        if record.get("flight_axes_active"):
            active += 1
        axes = record.get("axes")
        if isinstance(axes, dict):
            for axis, value in axes.items():
                number = _finite(value)
                if number is not None:
                    per_axis.setdefault(str(axis), []).append(int(number))
    return {
        "samples": len(records),
        "moved_samples": moved,
        "flight_axis_active_samples": active,
        "axis_span": {axis: max(values) - min(values) for axis, values in sorted(per_axis.items())},
    }


# --------------------------------------------------------------------------
# localization
# --------------------------------------------------------------------------
def _state_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    states: dict[str, int] = {}
    pose_status: dict[str, int] = {}
    successes = 0
    lost_episodes = 0
    previous_state = ""
    for record in records:
        state = str(record.get("next_mode") or record.get("mode") or "")
        if state:
            states[state] = states.get(state, 0) + 1
            if state == "LOST" and previous_state != "LOST":
                lost_episodes += 1
            previous_state = state
        if record.get("success"):
            successes += 1
        status = record.get("pose_status")
        if status:
            pose_status[str(status)] = pose_status.get(str(status), 0) + 1
    return {
        "successes": successes,
        "states": dict(sorted(states.items())),
        "pose_status": dict(sorted(pose_status.items())),
        "lost_episodes": lost_episodes,
    }


def _pairing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """How many frames actually carried a usable IMU sample into the worker."""
    paired = 0
    with_velocity = 0
    stale = 0
    errors: list[float] = []
    for record in records:
        telemetry_mono = _finite(record.get("fused_telemetry_mono"))
        if telemetry_mono is None:
            continue
        paired += 1
        if _finite(record.get("fused_speed_north")) is not None:
            with_velocity += 1
        capture_mono = _finite(record.get("source_frame_stamp_mono"))
        if capture_mono is None:
            continue
        error = abs(telemetry_mono - capture_mono)
        errors.append(error)
        if error > MAX_FUSED_SYNC_ERROR_S:
            stale += 1
    return {
        "fused_paired": paired,
        "fused_with_velocity": with_velocity,
        "sync_error_p50_s": _percentile(errors, 0.5),
        "sync_error_p95_s": _percentile(errors, 0.95),
        "sync_error_over_bound": stale,
    }


def _esekf_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = 0
    updates_seen = 0
    updates_accepted = 0
    d2: list[float] = []
    trace: list[float] = []
    for record in records:
        if str(record.get("prediction_mode") or "") == "esekf":
            predictions += 1
        accepted = record.get("esekf_update_accepted")
        if accepted is not None:
            updates_seen += 1
            updates_accepted += bool(accepted)
        value = _finite(record.get("esekf_d2"))
        if value is not None:
            d2.append(value)
        value = _finite(record.get("esekf_pos_trace"))
        if value is not None:
            trace.append(value)
    return {
        "esekf_predictions": predictions,
        "esekf_updates_seen": updates_seen,
        "esekf_updates_accepted": updates_accepted,
        "esekf_d2_p95": _percentile(d2, 0.95),
        "esekf_pos_trace_p95": _percentile(trace, 0.95),
    }


def summarize_localization(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frames": len(records),
        **_state_summary(records),
        **_pairing_summary(records),
        **_esekf_summary(records),
    }


def summarize_map_scale(
    fused: list[dict[str, Any]], localization: list[dict[str, Any]]
) -> dict[str, Any]:
    """Metres-per-map-unit from trajectory + IMU speed (informational).

    Never a verdict gate: a parked session simply reports unconverged. Any
    unexpected shape degrades to ``available=False`` so the report itself
    cannot crash on a new recorder version.
    """
    if estimate_map_scale is None:
        return {"available": False, "reason": "map_scale estimator unavailable"}
    speeds: list[tuple[float, float]] = []
    for row in fused:
        stamp = _finite(row.get("speed_mono_ns"))
        velocity = [
            _finite(row.get(key)) for key in ("speed_north_mps", "speed_east_mps", "speed_down_mps")
        ]
        if stamp is None or not all(value is not None for value in velocity):
            continue
        speeds.append(
            (stamp * 1e-9, math.sqrt(sum(value * value for value in velocity)))  # type: ignore[operator]
        )
    visual: list[tuple[float, float, float, float]] = []
    for row in localization:
        pose = row.get("pose")
        if (
            row.get("direct_status") not in {"FAST_TRACK", "RELOC_SEED"}
            or row.get("confidence_low")
            or row.get("imu_bridge")
            or row.get("reseed_confirming")
        ):
            continue
        stamp = _finite(row.get("source_frame_stamp_mono"))
        xyz = (
            [_finite(pose.get(axis)) for axis in ("x", "y", "z")]
            if isinstance(pose, dict)
            else [None, None, None]
        )
        if stamp is None or not all(value is not None for value in xyz):
            continue
        visual.append((stamp, xyz[0], xyz[1], xyz[2]))  # type: ignore[misc]
    try:
        estimate = estimate_map_scale(visual, speeds)
    except Exception as exc:  # defensive: report must not crash on data shape
        return {"available": False, "reason": f"estimator failed: {exc!r}"}
    return {
        "available": True,
        "converged": estimate.converged,
        "scale_m_per_unit": estimate.scale_m_per_unit,
        "q1_m_per_unit": estimate.q1_m_per_unit,
        "q3_m_per_unit": estimate.q3_m_per_unit,
        "n_segments": estimate.n_segments,
        "metric_distance_m": estimate.metric_distance_m,
        "visual_distance_u": estimate.visual_distance_u,
        "duration_s": estimate.duration_s,
        "reason": estimate.reason,
        "drift_ratio": estimate.drift_ratio,
    }


def summarize_frames(session: Path, errors: list[str] | None = None) -> dict[str, Any]:
    directory = session / "imu_test"
    rows = list(read_jsonl(directory / "frames.jsonl", errors))
    summary_path = directory / "summary.json"
    recorder: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                recorder = loaded
        except ValueError:
            pass
    stamps = sorted(
        stamp
        for stamp in (_finite(row.get("capture_stamp_mono")) for row in rows)
        if stamp is not None
    )
    with_velocity = sum(1 for row in rows if _finite(row.get("fused_speed_north")) is not None)
    missing = 0
    for row in rows:
        image = directory / str(row.get("file", ""))
        if not image.is_file() or (
            row.get("bytes") is not None and image.stat().st_size != row["bytes"]
        ):
            missing += 1
    return {
        "indexed": len(rows),
        "on_disk": (
            len(list((directory / "frames").glob("*.jpg")))
            if (directory / "frames").is_dir()
            else 0
        ),
        "with_velocity": with_velocity,
        "missing_or_truncated": missing,
        "rate_hz": _rate_hz(stamps),
        "recorder": recorder,
    }


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------
def _blockers(imu: dict[str, Any], localization: dict[str, Any]) -> list[str]:
    """Reasons the recording cannot answer the question at all."""
    found: list[str] = []
    if not imu["with_velocity"]:
        found.append("telemetry 沒有任何 NED 速度樣本，無法評估含飛控速度的鬆耦合融合。")
    if not imu["with_attitude"]:
        found.append("telemetry 沒有飛控融合姿態樣本。")
    if not localization["frames"]:
        found.append("localization.jsonl 沒有任何一幀；這段沒有開定位。")
    elif not localization["fused_paired"]:
        found.append("沒有任何一幀掛到 fused telemetry；IMU 從來沒有進到定位 worker。")
    return found


def _motion_warnings(
    imu: dict[str, Any], sticks: dict[str, Any], *, auto_recorded=False
) -> list[str]:
    """The aircraft has to have moved, and the pilot has to have moved it."""
    found: list[str] = []
    attitude_span = max(imu["roll_span_deg"], imu["pitch_span_deg"], imu["yaw_span_deg"])
    if imu["with_attitude"] and attitude_span < MIN_ATTITUDE_SPAN_DEG:
        found.append(
            f"姿態只動了 {attitude_span:.1f} 度；這段幾乎是靜止的，"
            "資訊量不足以判斷 IMU 有沒有幫助。"
        )
    if imu["with_velocity"] and imu["ground_speed_max_mps"] < MIN_SPEED_SPAN_MPS:
        found.append(
            f"最大地速只有 {imu['ground_speed_max_mps']:.2f} m/s；移動尺度及動態覆蓋有限。"
        )
    if auto_recorded:
        return found
    if not sticks["samples"]:
        found.append(
            "沒有 stick_axes；搖桿監控沒起來（直連無人機 Wi-Fi 就會這樣），"
            "IMU 變化沒有對應的操作輸入可以對照。"
        )
    elif not sticks["flight_axis_active_samples"]:
        found.append("搖桿全程沒有離開死區；這段不是手動飛的。")
    return found


def _recorder_warnings(frames: dict[str, Any]) -> list[str]:
    found: list[str] = []
    recorder = frames.get("recorder") or {}
    if isinstance(recorder, dict):
        if recorder.get("stop_reason"):
            found.append(f"錄製提早停止：{recorder['stop_reason']}；後續影格未必完整。")
        if recorder.get("incomplete") is True:
            found.append("錄製器 summary 標記 incomplete；這段資料不能當成完整場次。")
        if recorder.get("writer_shutdown_timeout") is True:
            found.append("錄製器 writer 在關閉期限內未完成；最後一段資料可能尚未寫入。")
        if recorder.get("writer_alive") is True:
            found.append("錄製器 summary 顯示 writer 仍在執行；影像索引可能尚未收尾。")
        writer_error = recorder.get("writer_error") or recorder.get("last_error")
        if writer_error:
            found.append(f"錄製器 writer 發生錯誤：{writer_error}")
    return found


def _coverage_warnings(localization: dict[str, Any], frames: dict[str, Any]) -> list[str]:
    """Gaps that leave part of the comparison unevaluated."""
    found: list[str] = []
    found.extend(_recorder_warnings(frames))
    if localization["lost_episodes"] < MIN_LOST_EPISODES:
        found.append("整段沒有進過 LOST；本段無法評估失鎖恢復，不需為收集資料刻意製造失鎖。")
    if localization["sync_error_over_bound"]:
        found.append(
            f"{localization['sync_error_over_bound']} 幀的 telemetry 比影像舊超過 "
            f"{MAX_FUSED_SYNC_ERROR_S}s；這些幀的 IMU 會被 protocol 擋掉。"
        )
    if localization["esekf_updates_seen"] and not localization["esekf_predictions"]:
        found.append(
            "ESEKF 全程沒有 arm（prediction_mode 從來不是 esekf）。"
            "可能是速度沒餵到、EKF 沒收斂，或飛行段太短。"
        )
    if not frames["indexed"]:
        found.append("沒有同步影格索引；只能分析既有姿態與遙測，不能重跑影像定位。")
    if frames.get("missing_or_truncated"):
        found.append(f"索引中的 {frames['missing_or_truncated']} 張影像缺失或大小不符。")
    elif frames["indexed"] and frames["on_disk"] < frames["indexed"]:
        found.append(f"frames.jsonl 有 {frames['indexed']} 筆但磁碟只有 {frames['on_disk']} 張圖。")
    dropped = frames.get("recorder", {}).get("dropped_queue_full")
    if dropped:
        found.append(f"錄製佇列滿而丟掉 {dropped} 張畫面（UI 優先，屬預期）。")
    return found


def verdict(
    imu: dict[str, Any],
    sticks: dict[str, Any],
    localization: dict[str, Any],
    frames: dict[str, Any],
    *,
    auto_recorded=False,
) -> tuple[str, list[str]]:
    blockers = _blockers(imu, localization)
    warnings = _motion_warnings(imu, sticks, auto_recorded=auto_recorded) + _coverage_warnings(
        localization, frames
    )
    if blockers:
        return "UNUSABLE", blockers + warnings
    if warnings:
        return "USABLE WITH GAPS", warnings
    return "USABLE", []


# --------------------------------------------------------------------------
def summarize_tick_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Where the Tk event loop spent its time (runbook 0b).

    runbook 0b measured ui_poll_delay_ms p50 at 25 ms on the *simulated* path
    and is explicit that the real-hardware number must be taken separately,
    because real frames are read on a dedicated RTSP worker thread instead of
    inline on Tk. This separate measurement shows which stage holds the loop,
    rather than choosing blind between the doc's candidates.
    """
    if not records:
        return {"windows": 0, "ticks": 0, "stages": {}, "worst": None}
    stages: dict[str, list[float]] = {}
    ticks = 0
    for record in records:
        ticks += int(record.get("ticks") or 0)
        for name, value in (record.get("stages") or {}).items():
            p50 = _finite(value.get("p50")) if isinstance(value, dict) else None
            if p50 is not None:
                stages.setdefault(name, []).append(p50)
    summary = {
        name: {
            "p50_ms": _percentile(values, 0.5),
            "max_ms": max(values),
        }
        for name, values in sorted(stages.items())
    }
    ranked = [
        (data["p50_ms"], name)
        for name, data in summary.items()
        if name != "_total" and data["p50_ms"] is not None
    ]
    return {
        "windows": len(records),
        "ticks": ticks,
        "tick_period_ms": records[-1].get("tick_period_ms"),
        "stages": summary,
        "worst": max(ranked)[1] if ranked else None,
    }


def summarize_ekf_capture(session, imu_rows, loc_rows, frames):
    """Report replay prerequisites, without mistaking collection for approval."""
    path = session / "imu_test/capture_contract.json"
    try:
        contract = json.loads(path.read_text())
    except (OSError, ValueError):
        contract = {}
    source_stamps = {}
    for name, field in (
        ("attitude", "attitude_mono_ns"),
        ("speed", "speed_mono_ns"),
        ("altitude", "altitude_mono_ns"),
        ("gimbal", "gimbal_mono_ns"),
    ):
        stamps = sorted(
            {
                float(r[field]) / 1e9
                for r in imu_rows
                if _finite(r.get(field)) is not None and float(r[field]) > 0
            }
        )
        source_stamps[name] = {
            "distinct_samples": len(stamps),
            "rate_hz": _rate_hz(stamps),
            "largest_gap_s": _largest_gap_s(stamps),
        }
    pairing = {}
    for name in ("attitude", "speed", "altitude"):
        offsets = []
        for row in loc_rows:
            capture = _finite(row.get("source_frame_stamp_mono"))
            stamp = _finite(row.get(f"fused_{name}_mono"))
            if capture is not None and stamp is not None:
                offsets.append(abs(capture - stamp))
        pairing[name] = {"paired": len(offsets), "abs_offset_p95_s": _percentile(offsets, 0.95)}

    def key(seq, stamp):
        return (seq, round(stamp, 6)) if seq is not None and stamp is not None else None

    pose_keys = {
        key(r.get("display_seq"), _finite(r.get("source_frame_stamp_mono"))) for r in loc_rows
    }
    pose_keys.discard(None)
    frame_rows = list(read_jsonl(session / "imu_test/frames.jsonl"))
    matched = sum(
        key(r.get("seq"), _finite(r.get("capture_stamp_mono"))) in pose_keys for r in frame_rows
    )
    camera = contract.get("active_query_camera") or {}
    expected_size = (camera.get("width"), camera.get("height"))
    dimensions_match = bool(
        all(expected_size)
        and frame_rows
        and all(
            (row.get("image_width"), row.get("image_height")) == expected_size for row in frame_rows
        )
    )
    timed = all(
        source_stamps[name]["distinct_samples"] >= 2 for name in ("attitude", "speed", "altitude")
    )
    visual_count = 0
    for row in loc_rows:
        pose = row.get("pose")
        xyz = [pose.get(k) for k in ("x", "y", "z")] if isinstance(pose, dict) else pose
        visual_count += bool(
            row.get("success") is not False
            and isinstance(xyz, (list, tuple))
            and len(xyz) >= 3
            and all(_finite(v) is not None for v in xyz[:3])
        )
    valid_telemetry = all(
        any(all(_finite(row.get(k)) is not None for k in keys) for row in imu_rows)
        for keys in (
            ("att_roll", "att_pitch", "att_yaw"),
            ("speed_north_mps", "speed_east_mps", "speed_down_mps"),
            ("drone_altitude_m",),
        )
    )
    pose_replay = bool(visual_count and timed and valid_telemetry)
    image_replay = bool(
        pose_replay
        and matched
        and not frames["missing_or_truncated"]
        and contract.get("intrinsics_match_active_pinhole")
        and dimensions_match
    )
    return {
        "pose_telemetry_replay_ready": pose_replay,
        "image_replay_ready": image_replay,
        "matched_recorded_frames": matched,
        "recorded_frames_without_result": len(frame_rows) - matched,
        "dimensions_match_active_camera": dimensions_match,
        "source_stamps": source_stamps,
        "pairing_by_source": pairing,
        "capture_contract": contract,
        "formal_control_ready": False,
        "remaining_before_formal_control": [
            "verify camera distortion domain and camera/body orientation convention",
            "independently verify metric scale, NED-to-map rotation and height datum",
            "calibrate measurement noise; evaluate on independent reference/holdout data",
            "offline A/B, fault injection, shadow-mode timing and operator-controlled validation",
        ],
    }


def build_report(session: Path) -> dict[str, Any]:
    errors = []
    archive = session / "trajectory.jsonl"
    telemetry_path = session / "telemetry.jsonl"
    localization_path = session / "localization.jsonl"
    telemetry = list(read_jsonl(telemetry_path if telemetry_path.is_file() else archive, errors))
    imu_rows = [r for r in telemetry if r.get("event") == "fused_odometry"]
    imu = summarize_imu(imu_rows)
    sticks = summarize_sticks([r for r in telemetry if r.get("event") == "stick_axes"])
    loc_events = list(
        read_jsonl(localization_path if localization_path.is_file() else archive, errors)
    )
    loc_rows = [r for r in loc_events if r.get("event") == "pose_result"]
    localization = summarize_localization(loc_rows)
    map_scale = summarize_map_scale(imu_rows, loc_rows)
    frames = summarize_frames(session, errors)
    tick_profile = summarize_tick_profile(
        [r for r in telemetry if r.get("event") == "ui_tick_profile"]
    )
    status, notes = verdict(
        imu,
        sticks,
        localization,
        frames,
        auto_recorded=any(r.get("event") == "auto_route_plan" for r in loc_events),
    )
    if errors:
        notes.extend(errors)
        if status == "USABLE":
            status = "USABLE WITH GAPS"
    return {
        "session": str(session),
        "imu": imu,
        "sticks": sticks,
        "localization": localization,
        "frames": frames,
        "map_scale": map_scale,
        "tick_profile": tick_profile,
        "verdict": status,
        "notes": notes,
        "integrity_errors": errors,
        "ekf_capture": summarize_ekf_capture(session, imu_rows, loc_rows, frames),
    }


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_capture(report: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    loc = report["localization"]
    capture = report.get("ekf_capture")
    if capture:
        lines += [
            "",
            "## EKF 資料準備（不等於正式控制驗收）",
            f"- 姿態＋遙測可重播：{capture['pose_telemetry_replay_ready']}",
            f"- 影像＋定位結果可配對：{capture['image_replay_ready']}；"
            f"已配對 {capture['matched_recorded_frames']} 張",
            (
                "- 紀錄包含 EKF 診斷；仍需獨立驗證效果。"
                if loc["esekf_updates_seen"]
                else "- 目前未啟用 EKF；沒有 EKF 協方差或創新量並非收集失敗。"
            ),
            "- 相機中心作導航位置是操作員指定的近似；不把相機與機身朝向設成相同。",
            "- 正式控制尚須：尺度／座標與畸變驗證、量測噪聲、獨立參考評估及旁路測試。",
        ]
        for name, data in capture["source_stamps"].items():
            lines.append(
                f"- {name}：{data['distinct_samples']} 個獨立時間戳，"
                f"來源更新率 {_number(data['rate_hz'], 2)} Hz"
            )
        for name, data in capture["pairing_by_source"].items():
            lines.append(f"- 影像與 {name} 時間差 p95 {_number(data['abs_offset_p95_s'])} s")
        for issue in capture["capture_contract"].get("issues", []):
            lines.append(f"- 標定紀錄問題：{issue}")
    return lines


def render(report: dict[str, Any]) -> str:
    imu = report["imu"]
    sticks = report["sticks"]
    loc = report["localization"]
    frames = report["frames"]
    lines = [
        f"# IMU 飛行測試報告 — {Path(report['session']).name}",
        "",
        f"判定：**{report['verdict']}**",
        "",
        "## IMU（telemetry.jsonl / fused_odometry）",
        f"- 樣本 {imu['samples']}，姿態 {imu['with_attitude']}，速度 {imu['with_velocity']}",
        f"- 取樣率 {_number(imu['rate_hz'], 2)} Hz，涵蓋 {_number(imu['duration_s'], 1)} s，"
        f"最大間隔 {_number(imu['largest_gap_s'])} s",
        f"- 姿態變化 roll {_number(imu['roll_span_deg'], 1)}° / "
        f"pitch {_number(imu['pitch_span_deg'], 1)}° / yaw {_number(imu['yaw_span_deg'], 1)}°",
        f"- 最大地速 {_number(imu['ground_speed_max_mps'], 2)} m/s，"
        f"最大垂直速 {_number(imu['vertical_speed_max_mps'], 2)} m/s",
        "",
        "## 搖桿（telemetry.jsonl / stick_axes）",
        f"- 樣本 {sticks['samples']}，有動作 {sticks['moved_samples']}，"
        f"離開死區 {sticks['flight_axis_active_samples']}",
        f"- 各軸行程 {sticks['axis_span'] or '-'}",
        "",
        "## 定位（localization.jsonl）",
        f"- 幀數 {loc['frames']}，成功 {loc['successes']}，LOST 段落 {loc['lost_episodes']}",
        f"- 狀態分布 {loc['states'] or '-'}",
        f"- pose_status {loc['pose_status'] or '-'}",
        f"- 掛到 IMU 的幀 {loc['fused_paired']}（其中帶速度 {loc['fused_with_velocity']}）",
        f"- 影像與 telemetry 時間差 p50 {_number(loc['sync_error_p50_s'])} s / "
        f"p95 {_number(loc['sync_error_p95_s'])} s，超過 {MAX_FUSED_SYNC_ERROR_S}s 的有 "
        f"{loc['sync_error_over_bound']} 幀",
        f"- ESEKF 預測 {loc['esekf_predictions']} 幀，視覺更新 "
        f"{loc['esekf_updates_accepted']}/{loc['esekf_updates_seen']} 接受，"
        f"d2 p95 {_number(loc['esekf_d2_p95'])}，位置協方差 trace p95 "
        f"{_number(loc['esekf_pos_trace_p95'])}",
        "",
        "## 畫面（imu_test/）",
        f"- 索引 {frames['indexed']} 筆，磁碟 {frames['on_disk']} 張，"
        f"帶速度 {frames['with_velocity']} 張，約 {_number(frames['rate_hz'], 2)} Hz",
    ]
    recorder = frames.get("recorder") or {}
    if recorder:
        lines.append(
            f"- 錄製器 written={recorder.get('written')} "
            f"dropped_queue_full={recorder.get('dropped_queue_full')} "
            f"stop_reason={recorder.get('stop_reason') or '-'} "
            f"incomplete={recorder.get('incomplete', '-')} "
            f"writer_alive={recorder.get('writer_alive', '-')} "
            f"shutdown_timeout={recorder.get('writer_shutdown_timeout', '-')}"
        )
        writer_error = recorder.get("writer_error") or recorder.get("last_error")
        if writer_error:
            lines.append(f"- 錄製器 writer error={writer_error}")
    lines += _render_capture(report)
    scale = report.get("map_scale") or {}
    if scale.get("available"):
        lines += [
            "",
            "## 地圖尺度（trajectory × IMU speed，非判定門檻）",
        ]
        if scale.get("converged"):
            lines.append(
                f"- {scale['n_segments']} 段中位數 {_number(scale.get('scale_m_per_unit'), 3)} m/unit"
                f"（IQR {_number(scale.get('q1_m_per_unit'), 3)}–{_number(scale.get('q3_m_per_unit'), 3)}），"
                f"共 {_number(scale.get('metric_distance_m'), 1)} m / "
                f"{_number(scale.get('visual_distance_u'), 2)} unit"
            )
            if scale.get("drift_ratio") is not None:
                lines.append(
                    f"- 前後半段比值 {_number(scale.get('drift_ratio'), 3)}"
                    "（遠離 1 表示視覺尺度在飄，不代表估計錯）"
                )
        else:
            lines.append(f"- 未收斂：{scale.get('reason') or '-'}")
    tick = report.get("tick_profile") or {}
    if tick.get("ticks"):
        total = (tick.get("stages") or {}).get("_total") or {}
        lines += [
            "",
            "## UI 事件圈（runbook 0b：真機數字）",
            f"- {tick['windows']} 個視窗 / {tick['ticks']} 個 tick，"
            f"tick 週期 {_number(tick.get('tick_period_ms'), 0)} ms",
            f"- 整個 tick p50 {_number(total.get('p50_ms'))} ms，"
            f"最慢 {_number(total.get('max_ms'))} ms",
            f"- 最貴的階段：{tick.get('worst') or '-'}",
        ]
        for name, data in (tick.get("stages") or {}).items():
            if name == "_total":
                continue
            lines.append(
                f"    {name}: p50 {_number(data.get('p50_ms'))} ms / "
                f"max {_number(data.get('max_ms'))} ms"
            )
    if report["notes"]:
        lines += ["", "## 要注意的"]
        lines += [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="flight_logs session 目錄")
    parser.add_argument("--out", default="", help="另外寫一份 markdown 到這個路徑")
    parser.add_argument("--json", action="store_true", help="輸出 JSON 而不是 markdown")
    parser.add_argument("--json-out", type=Path, help="另存可機讀的資料完整性報告")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = Path(args.session).expanduser().resolve()
    if not session.is_dir():
        raise SystemExit(f"session 目錄不存在：{session}")
    report = build_report(session)
    text = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.json
        else render(report)
    )
    print(text, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(report), encoding="utf-8")
        print(f"\n[報告] {out}")
    return 0 if report["verdict"] != "UNUSABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
