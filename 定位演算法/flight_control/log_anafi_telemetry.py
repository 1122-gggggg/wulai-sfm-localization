#!/usr/bin/env python3
"""Passive ANAFI / SkyController ARSDK-NG telemetry logger.

SAFETY
  - Connects only. Never TakeOff / PCMD / Landing / Emergency / moveBy.
  - Intended for lab logging while a human flies with SkyController sticks.
  - Use ONE ARSDK-NG control connection: prefer PC -> SkyController (USB or SC WiFi).

Network
  SkyController pilot (sticks) + PC ARSDK-NG both go through the SC:
    PC <-> SkyController (IP often 192.168.53.1) <-> ANAFI
  Do not also open a second aircraft control connection while SC owns the link.

Prerequisite (from repository root)
  bash tools/build_arsdk_ng.sh
  The pinned libraries under 執行環境/native/arsdk-ng/lib are loaded automatically;
  set ARSDK_NG_LIB_DIR to use another compatible build.

Examples (from repository root)
  .venv/bin/python 定位演算法/flight_control/log_anafi_telemetry.py \
    --ip 192.168.53.1 --controller skycontroller3 --secs 120
  .venv/bin/python 定位演算法/flight_control/log_anafi_telemetry.py \
    --ip 192.168.42.1 --controller drone --secs 60
  .venv/bin/python 定位演算法/flight_control/log_anafi_telemetry.py \
    --ip 192.168.53.1 --controller skycontroller3 --secs 60 --with-video
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the project's single ARSDK-NG connection and RTSP helper.
from rtsp_frame_source import RtspFrameGrabber, connect  # noqa: E402


def _safe_state(drone, msg) -> Any | None:
    try:
        return drone.get_state(msg)
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "name") and not callable(obj):
        try:
            return str(obj.name)
        except Exception:
            pass
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _build_message_table(client):
    """Resolve supported ARSDK message descriptors, skipping absent channels."""
    paths = (
        ("piloting.attitude", "Ardrone3.PilotingState.AttitudeChanged"),
        ("piloting.altitude", "Ardrone3.PilotingState.AltitudeChanged"),
        ("piloting.altitude_agl", "Ardrone3.PilotingState.AltitudeAboveGroundChanged"),
        ("piloting.position", "Ardrone3.PilotingState.PositionChanged"),
        ("piloting.speed", "Ardrone3.PilotingState.SpeedChanged"),
        ("piloting.airspeed", "Ardrone3.PilotingState.AirSpeedChanged"),
        ("piloting.gps_location", "Ardrone3.PilotingState.GpsLocationChanged"),
        ("piloting.flying_state", "Ardrone3.PilotingState.FlyingStateChanged"),
        ("piloting.motion_state", "Ardrone3.PilotingState.MotionState"),
        ("piloting.alert", "Ardrone3.PilotingState.AlertStateChanged"),
        ("piloting.vibration", "Ardrone3.PilotingState.VibrationLevelChanged"),
        ("piloting.wind", "Ardrone3.PilotingState.WindStateChanged"),
        ("piloting.rth_state", "Ardrone3.PilotingState.NavigateHomeStateChanged"),
        ("gps.fix", "Ardrone3.GPSSettingsState.GPSFixStateChanged"),
        ("gps.home", "Ardrone3.GPSSettingsState.HomeChanged"),
        ("common.battery", "Common.CommonState.BatteryStateChanged"),
        ("common.wifi", "Common.CommonState.WifiSignalChanged"),
        ("common.link", "Common.CommonState.LinkSignalQuality"),
        ("common.sensors", "Common.CommonState.SensorsStatesListChanged"),
        ("gimbal.attitude", "Gimbal.attitude"),
        ("gimbal.calibration", "Gimbal.calibration_state"),
        ("settings.max_altitude", "Ardrone3.PilotingSettingsState.MaxAltitudeChanged"),
        ("settings.max_distance", "Ardrone3.PilotingSettingsState.MaxDistanceChanged"),
        ("settings.max_tilt", "Ardrone3.PilotingSettingsState.MaxTiltChanged"),
        (
            "settings.nofly_maxdist",
            "Ardrone3.PilotingSettingsState.NoFlyOverMaxDistanceChanged",
        ),
        ("settings.max_vspeed", "Ardrone3.SpeedSettingsState.MaxVerticalSpeedChanged"),
        ("settings.max_yaw_rate", "Ardrone3.SpeedSettingsState.MaxRotationSpeedChanged"),
        (
            "settings.max_pitch_roll_rate",
            "Ardrone3.SpeedSettingsState.MaxPitchRollRotationSpeedChanged",
        ),
        ("camera.orientation_v2", "Ardrone3.CameraState.OrientationV2"),
    )
    table = []
    for label, path in paths:
        descriptor = client.messages
        try:
            for component in path.split("."):
                descriptor = getattr(descriptor, component)
        except AttributeError:
            continue
        table.append((label, descriptor))
    return table


def sample_once(drone, messages) -> dict:
    row: dict[str, Any] = {
        "t_unix": time.time(),
        "t_mono": time.monotonic(),
        "t_iso": datetime.now(timezone.utc).isoformat(),
    }
    present = []
    missing = []
    for label, msg in messages:
        st = _safe_state(drone, msg)
        if st is None:
            missing.append(label)
            continue
        present.append(label)
        row[label] = _jsonable(st)
    row["_channels_ok"] = present
    row["_channels_missing"] = missing
    return row


def _start_video_grabber(drone, *, enabled: bool):
    if not enabled:
        return None
    candidate = None
    try:
        candidate = RtspFrameGrabber(drone, resize=(1280, 720), stale_s=0.8)
        candidate.start()
        print("[log] video grabber started (still no arm)", flush=True)
        return candidate
    except Exception as exc:
        print(f"[log] video grabber failed (telemetry continues): {exc}", flush=True)
        if candidate is not None:
            try:
                candidate.stop()
            except Exception as stop_exc:
                print(f"[log] partial video cleanup failed: {stop_exc}", flush=True)
        return None


def _add_video_sample(row: dict[str, Any], grabber) -> tuple[int, int]:
    metadata = dict(grabber.stream_metadata)
    degraded = bool(metadata.get("timestamp_degraded", True))
    trusted = bool(metadata.get("source_timestamp_trusted", False))
    stamp_source = str(metadata.get("stamp_source", "unknown"))
    row["video.source_timestamp_degraded"] = degraded
    row["video.source_timestamp_trusted"] = trusted
    row["video.stamp_source"] = stamp_source
    try:
        result = grabber()
    except Exception:
        result = None
    if result is None:
        row["video.frame"] = None
        return 0, 1
    frame, stamp = result if isinstance(result, tuple) else (result, None)
    height, width = frame.shape[:2] if frame is not None else (None, None)
    row["video.frame"] = {
        "w": width,
        "h": height,
        "stamp_mono": stamp,
        "age_s": (time.monotonic() - stamp) if stamp else None,
        "timestamp_degraded": degraded,
        "source_timestamp_trusted": trusted,
        "stamp_source": stamp_source,
    }
    return 1, 0


def _report_first_sample(row: dict[str, Any], message_count: int) -> None:
    present = row.get("_channels_ok", [])
    print(
        f"[log] first sample channels present: {len(present)} / {message_count}",
        flush=True,
    )
    for label in present[:20]:
        print(f"  OK  {label}", flush=True)
    missing = row.get("_channels_missing", [])
    if missing:
        print(
            f"[log] missing on first sample ({len(missing)}): "
            f"{', '.join(missing[:12])}...",
            flush=True,
        )


def _report_progress(sample_count: int, row: dict[str, Any]) -> None:
    attitude = row.get("piloting.attitude")
    altitude = row.get("piloting.altitude")
    speed = row.get("piloting.speed")
    battery = row.get("common.battery") or row.get("battery.capacity")
    print(
        f"[log] n={sample_count} att={attitude} alt={altitude} "
        f"spd={speed} bat={battery}",
        flush=True,
    )


def _write_start_record(
    stream,
    *,
    ip: str,
    controller: str,
    secs: float,
    hz: float,
    with_video: bool,
    messages,
) -> None:
    record = {
        "event": "start",
        "ip": ip,
        "controller": controller,
        "secs": secs,
        "hz": hz,
        "with_video": bool(with_video),
        "channel_labels": [label for label, _ in messages],
        "safety": "passive_no_arm",
        "t_iso": datetime.now(timezone.utc).isoformat(),
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _write_samples(stream, *, drone, messages, grabber, secs: float, hz: float) -> tuple[int, int, int]:
    period = 1.0 / max(1e-3, hz)
    end_time = time.monotonic() + max(0.1, secs)
    sample_count = video_ok = video_none = 0
    first_sample = True
    while time.monotonic() < end_time:
        started_at = time.monotonic()
        row = sample_once(drone, messages)
        if grabber is not None:
            ok_increment, none_increment = _add_video_sample(row, grabber)
            video_ok += ok_increment
            video_none += none_increment
        if first_sample:
            _report_first_sample(row, len(messages))
            first_sample = False
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        sample_count += 1
        if sample_count % max(1, int(hz)) == 0:
            stream.flush()
            _report_progress(sample_count, row)
        elapsed = time.monotonic() - started_at
        time.sleep(max(0.0, period - elapsed))
    return sample_count, video_ok, video_none


def _write_end_record(stream, *, samples: int, video_ok: int, video_none: int) -> None:
    record = {
        "event": "end",
        "samples": samples,
        "video_frames_ok": video_ok,
        "video_frames_none": video_none,
        "t_iso": datetime.now(timezone.utc).isoformat(),
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cleanup_session(drone, grabber) -> None:
    if grabber is not None:
        try:
            grabber.stop()
        except Exception:
            pass
    try:
        drone.disconnect()
    except Exception:
        pass


def run_log(*, ip: str, controller: str, secs: float, hz: float, out: Path,
            with_video: bool) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[log] connecting ip={ip} controller={controller}", flush=True)
    print("[log] PASSIVE ONLY — no TakeOff/PCMD/Landing will be sent", flush=True)
    drone = connect(ip, controller=controller)
    messages = _build_message_table(drone)
    grabber = _start_video_grabber(drone, enabled=with_video)
    sample_count = video_ok = video_none = 0
    try:
        with out.open("w", encoding="utf-8") as stream:
            _write_start_record(
                stream,
                ip=ip,
                controller=controller,
                secs=secs,
                hz=hz,
                with_video=with_video,
                messages=messages,
            )
            sample_count, video_ok, video_none = _write_samples(
                stream,
                drone=drone,
                messages=messages,
                grabber=grabber,
                secs=secs,
                hz=hz,
            )
            _write_end_record(
                stream,
                samples=sample_count,
                video_ok=video_ok,
                video_none=video_none,
            )
    finally:
        _cleanup_session(drone, grabber)
    print(f"[log] wrote {sample_count} samples -> {out}", flush=True)
    return out


def summarize(path: Path) -> None:
    """Quick post-run channel coverage summary."""
    present_counts: dict[str, int] = {}
    samples = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("event"):
                continue
            samples += 1
            for lab in row.get("_channels_ok", []):
                present_counts[lab] = present_counts.get(lab, 0) + 1
    print(f"\n=== summary {path.name} samples={samples} ===")
    for lab, c in sorted(present_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {c:6d}/{samples}  {lab}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Passive ANAFI telemetry logger (no arm)")
    ap.add_argument("--ip", default="192.168.53.1",
                    help="192.168.53.1 SkyController / 192.168.42.1 direct drone")
    ap.add_argument("--controller", default="auto",
                    help="auto / skycontroller3 / drone")
    ap.add_argument("--secs", type=float, default=120.0)
    ap.add_argument("--hz", type=float, default=20.0,
                    help="sample rate (get_state polling); attitude often ~5 Hz source")
    ap.add_argument("--with-video", action="store_true",
                    help="also sample Parrot RTSP /live frames for fps/age (no arm)")
    ap.add_argument("--out", default="",
                    help="JSONL path; default under outputs/flight_logs/")
    ap.add_argument("--summarize", default="",
                    help="only summarize an existing JSONL and exit")
    args = ap.parse_args()

    if args.summarize:
        summarize(Path(args.summarize))
        return

    if args.out:
        out = Path(args.out)
    else:
        root = Path(__file__).resolve().parents[2]  # workspace/package root
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = root / "outputs" / "flight_logs" / f"telemetry_{stamp}.jsonl"

    run_log(ip=args.ip, controller=args.controller, secs=args.secs, hz=args.hz,
            out=out, with_video=args.with_video)
    summarize(out)


if __name__ == "__main__":
    main()
