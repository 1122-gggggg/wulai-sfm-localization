#!/usr/bin/env python3
"""Read-only first look at an IMU flight-test session.

Answers one question before any GPU time is spent: does this recording actually
contain what an IMU-aided localization evaluation needs? A trip that produced a
session with no velocity, no stick input, or no LOST episode cannot be rescued
offline, and the ESEKF A/B (``定位演算法/validation/benchmark_esekf_live_replay.py``)
would just come back ``INVALID``/``DORMANT`` after a full replay.

Reads only ``telemetry.jsonl``, ``localization.jsonl`` and ``imu_test/`` from
the session directory. Writes nothing unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator

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


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
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
        attitude = [
            _finite(record.get(key))
            for key in ("att_roll", "att_pitch", "att_yaw")
        ]
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
        "axis_span": {
            axis: max(values) - min(values) for axis, values in sorted(per_axis.items())
        },
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


def summarize_frames(session: Path) -> dict[str, Any]:
    directory = session / "imu_test"
    rows = list(read_jsonl(directory / "frames.jsonl"))
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
        stamp for stamp in (_finite(row.get("capture_stamp_mono")) for row in rows)
        if stamp is not None
    )
    with_velocity = sum(
        1 for row in rows if _finite(row.get("fused_speed_north")) is not None
    )
    return {
        "indexed": len(rows),
        "on_disk": (
            len(list((directory / "frames").glob("*.jpg")))
            if (directory / "frames").is_dir()
            else 0
        ),
        "with_velocity": with_velocity,
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
        found.append(
            "telemetry 沒有任何 NED 速度樣本；ESEKF 的 prediction_allowed() 永遠是 False，"
            "離線 A/B 只會回 INVALID。"
        )
    if not imu["with_attitude"]:
        found.append("telemetry 沒有姿態樣本；沒有 IMU 可評。")
    if not localization["frames"]:
        found.append("localization.jsonl 沒有任何一幀；這段沒有開定位。")
    elif not localization["fused_paired"]:
        found.append("沒有任何一幀掛到 fused telemetry；IMU 從來沒有進到定位 worker。")
    return found


def _motion_warnings(imu: dict[str, Any], sticks: dict[str, Any]) -> list[str]:
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
            f"最大地速只有 {imu['ground_speed_max_mps']:.2f} m/s；EKF 大概不會收斂。"
        )
    if not sticks["samples"]:
        found.append(
            "沒有 stick_axes；搖桿監控沒起來（直連無人機 Wi-Fi 就會這樣），"
            "IMU 變化沒有對應的操作輸入可以對照。"
        )
    elif not sticks["flight_axis_active_samples"]:
        found.append("搖桿全程沒有離開死區；這段不是手動飛的。")
    return found


def _coverage_warnings(
    localization: dict[str, Any], frames: dict[str, Any]
) -> list[str]:
    """Gaps that leave part of the comparison unevaluated."""
    found: list[str] = []
    if localization["lost_episodes"] < MIN_LOST_EPISODES:
        found.append(
            "整段沒有進過 LOST；recovery 這一半評不到，下次記得掃過難定位區再回來。"
        )
    if localization["sync_error_over_bound"]:
        found.append(
            f"{localization['sync_error_over_bound']} 幀的 telemetry 比影像舊超過 "
            f"{MAX_FUSED_SYNC_ERROR_S}s；這些幀的 IMU 會被 protocol 擋掉。"
        )
    if localization["frames"] and not localization["esekf_predictions"]:
        found.append(
            "ESEKF 全程沒有 arm（prediction_mode 從來不是 esekf）。"
            "可能是速度沒餵到、EKF 沒收斂，或飛行段太短。"
        )
    if frames["indexed"] and frames["on_disk"] < frames["indexed"]:
        found.append(
            f"frames.jsonl 有 {frames['indexed']} 筆但磁碟只有 {frames['on_disk']} 張圖。"
        )
    dropped = frames.get("recorder", {}).get("dropped_queue_full")
    if dropped:
        found.append(f"錄製佇列滿而丟掉 {dropped} 張畫面（UI 優先，屬預期）。")
    return found


def verdict(
    imu: dict[str, Any],
    sticks: dict[str, Any],
    localization: dict[str, Any],
    frames: dict[str, Any],
) -> tuple[str, list[str]]:
    blockers = _blockers(imu, localization)
    warnings = _motion_warnings(imu, sticks) + _coverage_warnings(localization, frames)
    if blockers:
        return "UNUSABLE", blockers + warnings
    if warnings:
        return "USABLE WITH GAPS", warnings
    return "USABLE", []


# --------------------------------------------------------------------------
def build_report(session: Path) -> dict[str, Any]:
    telemetry = list(read_jsonl(session / "telemetry.jsonl"))
    imu = summarize_imu([r for r in telemetry if r.get("event") == "fused_odometry"])
    sticks = summarize_sticks([r for r in telemetry if r.get("event") == "stick_axes"])
    localization = summarize_localization(list(read_jsonl(session / "localization.jsonl")))
    frames = summarize_frames(session)
    status, notes = verdict(imu, sticks, localization, frames)
    return {
        "session": str(session),
        "imu": imu,
        "sticks": sticks,
        "localization": localization,
        "frames": frames,
        "verdict": status,
        "notes": notes,
    }


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


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
            f"stop_reason={recorder.get('stop_reason') or '-'}"
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
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render(report), encoding="utf-8")
        print(f"\n[報告] {out}")
    return 0 if report["verdict"] != "UNUSABLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
