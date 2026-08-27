#!/usr/bin/env python3
"""Passive flight session: log everything + live-localize. NEVER arms.

Use while a human pilot flies with SkyController sticks. PC only observes:
  - full Olympe telemetry (attitude/alt/speed/gps/battery/gimbal/...)
  - PDRAW video frame health (shape/age/fps)
  - production localizer pose + inliers/mode/weak (algorithm validation)

SAFETY
  - No TakeOff / PCMD / Landing / Emergency / moveBy / gimbal set.
  - Single Olympe connection via SkyController (default 192.168.53.1).

Example
  python passive_flight_session.py --ip 192.168.53.1 --controller skycontroller3 \\
      --secs 900 --hz 10
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# flight_control is cwd / on sys.path when launched from there
import log_anafi_telemetry as tel
import olympe_frame_source as ofs
import path_follow_flight as pff
import real_path_follow_controller as rpf


def _jsonable(obj: Any) -> Any:
    return tel._jsonable(obj)


def _finite_pose(p) -> bool:
    if p is None:
        return False
    return all(math.isfinite(float(v)) for v in (p.x, p.y, p.z))


@dataclass
class _SessionResources:
    grabber: Any = None
    localizer: Any = None
    waypoints: Any = None
    cumulative_path: Any = None
    path_length: float | None = None


@dataclass
class _SessionStats:
    samples: int = 0
    video_ok: int = 0
    video_none: int = 0
    loc_ok: int = 0
    loc_fail: int = 0
    loc_skip: int = 0
    jump_rejects: int = 0
    last_good_xyz: tuple[float, float, float] | None = None


def _install_signal_handlers(stop: dict[str, bool]) -> None:
    def request_stop(*_args) -> None:
        stop["f"] = True
        print("[session] stop requested (signal)", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def _load_route(resources: _SessionResources) -> None:
    try:
        map_frame = (
            rpf.load_map_frame(pff.MAP_ALIGN)
            if pff.MAP_ALIGN
            else rpf.LEGACY_MAP_FRAME
        )
        resources.waypoints = rpf.load_waypoints(pff.PATH_JSON, map_frame=map_frame)
        poles = rpf.load_poles(pff.POLES_JSON, map_frame)
        controller = rpf.RouteAutoController(
            resources.waypoints,
            poles,
            rpf.config_for_route(
                pff.PATH_JSON,
                rpf.ControlConfig(map_frame=map_frame),
            ),
        )
        resources.cumulative_path = controller.cum
        resources.path_length = float(controller.path_len)
        print(
            f"[session] route loaded: {len(resources.waypoints)} wp, "
            f"len={resources.path_length:.2f}u  poles={len(poles)}",
            flush=True,
        )
    except Exception as exc:
        print(f"[session] route load failed (loc continues): {exc}", flush=True)


def _start_resources(drone, *, with_localize: bool) -> _SessionResources:
    resources = _SessionResources()
    try:
        resources.grabber = ofs.OlympePdrawGrabber(
            drone,
            resize=(1280, 720),
            stale_s=0.8,
        ).start()
        print("[session] video grabber started", flush=True)
        if with_localize:
            print(
                "[session] loading localizer models (stay on ground until ready)...",
                flush=True,
            )
            resources.localizer = pff.build_localizer(resources.grabber)
            resources.localizer.ensure_models()
            _load_route(resources)
            print("[session] localizer ready — pilot may take off with sticks", flush=True)
    except Exception as exc:
        print(f"[session] video/localizer setup failed: {exc}", flush=True)
        traceback.print_exc()
        if resources.grabber is not None:
            try:
                resources.grabber.stop()
            except Exception:
                pass
            resources.grabber = None
        resources.localizer = None
    return resources


def _max_pose_jump() -> float:
    try:
        return pff._env_float(
            "SFM_MAX_POSE_JUMP_U",
            1.5,
            minimum=0.1,
            maximum=50.0,
        )
    except Exception:
        return 1.5


def _add_video_sample(
    row: dict[str, Any],
    resources: _SessionResources,
    stats: _SessionStats,
) -> None:
    grabber = resources.grabber
    if grabber is None:
        row["video.frame"] = None
        return
    try:
        result = grabber()
    except Exception:
        result = None
    if result is None:
        stats.video_none += 1
        row["video.frame"] = None
        row["video.healthy"] = bool(grabber.is_healthy())
        return
    stats.video_ok += 1
    frame, stamp = result if isinstance(result, tuple) else (result, None)
    height, width = frame.shape[:2] if frame is not None else (None, None)
    row["video.frame"] = {
        "w": width,
        "h": height,
        "stamp_mono": stamp,
        "age_s": (time.monotonic() - stamp) if stamp else None,
    }
    row["video.healthy"] = bool(grabber.is_healthy())
    row["video.fps"] = float(getattr(grabber, "fps", 0.0) or 0.0)


def _pose_path_metrics(pose, resources: _SessionResources) -> tuple[float | None, float | None]:
    if resources.waypoints is None or resources.cumulative_path is None:
        return None, None
    try:
        distance, _segments, _segment, progress = rpf.project_to_path(
            [pose.x, pose.y, pose.z],
            resources.waypoints,
            resources.cumulative_path,
        )
        fraction = progress / resources.path_length if resources.path_length else None
        return float(distance), float(fraction) if fraction is not None else None
    except Exception:
        return None, None


def _pose_is_jump(pose, stats: _SessionStats, max_jump_u: float) -> bool:
    if stats.last_good_xyz is None:
        return False
    dx = pose.x - stats.last_good_xyz[0]
    dy = pose.y - stats.last_good_xyz[1]
    dz = pose.z - stats.last_good_xyz[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) > max_jump_u


def _record_pose_quality(pose, stats: _SessionStats, max_jump_u: float) -> tuple[bool, bool]:
    finite = _finite_pose(pose)
    jump = finite and _pose_is_jump(pose, stats, max_jump_u)
    if jump:
        stats.jump_rejects += 1
    if finite and not jump:
        stats.last_good_xyz = (float(pose.x), float(pose.y), float(pose.z))
        stats.loc_ok += 1
    else:
        stats.loc_fail += 1
    return finite, jump


def _localization_record(
    pose,
    info: dict[str, Any],
    *,
    finite: bool,
    jump: bool,
    resources: _SessionResources,
) -> dict[str, Any]:
    path_error, progress = (
        _pose_path_metrics(pose, resources) if finite else (None, None)
    )
    return {
        "ok": bool(finite and not jump),
        "finite": finite,
        "jump_reject": jump,
        "x": float(pose.x) if finite else None,
        "y": float(pose.y) if finite else None,
        "z": float(pose.z) if finite else None,
        "yaw": float(getattr(pose, "yaw", float("nan"))) if finite else None,
        "stamp": float(getattr(pose, "stamp", 0.0) or 0.0) if finite else None,
        "mode": info.get("mode") or info.get("next_mode"),
        "inliers": int(info.get("inliers", 0) or 0),
        "weak": bool(info.get("weak", False)),
        "reproj": info.get("reproj") or info.get("reproj_rms"),
        "path_error_u": path_error,
        "progress": progress,
    }


def _add_localization_sample(
    row: dict[str, Any],
    resources: _SessionResources,
    stats: _SessionStats,
    *,
    max_jump_u: float,
) -> None:
    localizer = resources.localizer
    if localizer is None:
        stats.loc_skip += 1
        row["loc"] = None
        return
    try:
        pose = localizer.get_pose()
        info = dict(getattr(localizer, "last_info", {}) or {})
        finite, jump = _record_pose_quality(pose, stats, max_jump_u)
        row["loc"] = _localization_record(
            pose,
            info,
            finite=finite,
            jump=jump,
            resources=resources,
        )
    except Exception as exc:
        stats.loc_fail += 1
        row["loc"] = {"ok": False, "error": repr(exc)}


def _report_progress(stats: _SessionStats, row: dict[str, Any]) -> None:
    altitude = row.get("piloting.altitude")
    flying = row.get("piloting.flying_state")
    battery = row.get("common.battery") or row.get("battery.capacity")
    localization = row.get("loc") or {}
    print(
        f"[session] n={stats.samples} fly={flying} alt={altitude} bat={battery} "
        f"vid_ok={stats.video_ok} loc_ok={stats.loc_ok}/{stats.loc_ok + stats.loc_fail} "
        f"mode={localization.get('mode')} inl={localization.get('inliers')} "
        f"pos=({localization.get('x')},{localization.get('y')},{localization.get('z')}) "
        f"path_err={localization.get('path_error_u')}",
        flush=True,
    )


def _write_start_record(
    stream,
    *,
    ip: str,
    controller: str,
    secs: float,
    hz: float,
    with_localize: bool,
    resources: _SessionResources,
    messages,
    max_jump_u: float,
) -> None:
    record = {
        "event": "start",
        "ip": ip,
        "controller": controller,
        "secs": secs,
        "hz": hz,
        "with_localize": bool(with_localize and resources.localizer is not None),
        "bundle": str(pff.XBUN),
        "path_json": str(pff.PATH_JSON),
        "channel_labels": [label for label, _ in messages],
        "safety": "passive_no_arm",
        "max_pose_jump_u": max_jump_u,
        "t_iso": datetime.now(timezone.utc).isoformat(),
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def _write_samples(
    stream,
    *,
    drone,
    messages,
    resources: _SessionResources,
    stop: dict[str, bool],
    secs: float,
    hz: float,
    max_jump_u: float,
) -> _SessionStats:
    stats = _SessionStats()
    period = 1.0 / max(1e-3, hz)
    end_time = time.monotonic() + max(0.1, secs)
    while not stop["f"] and time.monotonic() < end_time:
        started_at = time.monotonic()
        row = tel.sample_once(drone, messages)
        _add_video_sample(row, resources, stats)
        _add_localization_sample(
            row,
            resources,
            stats,
            max_jump_u=max_jump_u,
        )
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats.samples += 1
        if stats.samples % max(1, int(hz)) == 0:
            stream.flush()
            _report_progress(stats, row)
        elapsed = time.monotonic() - started_at
        time.sleep(max(0.0, period - elapsed))
    return stats


def _write_end_record(stream, stats: _SessionStats, *, stopped: bool) -> None:
    record = {
        "event": "end",
        "samples": stats.samples,
        "video_frames_ok": stats.video_ok,
        "video_frames_none": stats.video_none,
        "loc_ok": stats.loc_ok,
        "loc_fail": stats.loc_fail,
        "loc_skip": stats.loc_skip,
        "jump_rejects": stats.jump_rejects,
        "stopped": stopped,
        "t_iso": datetime.now(timezone.utc).isoformat(),
    }
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cleanup_resources(drone, resources: _SessionResources) -> None:
    if resources.grabber is not None:
        try:
            resources.grabber.stop()
        except Exception:
            pass
    try:
        drone.disconnect()
    except Exception:
        pass


def run_session(*, ip: str, controller: str, secs: float, hz: float,
                out: Path, with_localize: bool) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    messages = tel._build_message_table()
    stop = {"f": False}
    _install_signal_handlers(stop)
    print(f"[session] connecting ip={ip} controller={controller}", flush=True)
    print("[session] PASSIVE ONLY — no TakeOff/PCMD/Landing/Emergency/gimbal cmd", flush=True)
    drone = ofs.connect(ip, controller=controller)
    resources = _start_resources(drone, with_localize=with_localize)
    max_jump_u = _max_pose_jump()
    stats = _SessionStats()
    try:
        with out.open("w", encoding="utf-8") as stream:
            _write_start_record(
                stream,
                ip=ip,
                controller=controller,
                secs=secs,
                hz=hz,
                with_localize=with_localize,
                resources=resources,
                messages=messages,
                max_jump_u=max_jump_u,
            )
            stats = _write_samples(
                stream,
                drone=drone,
                messages=messages,
                resources=resources,
                stop=stop,
                secs=secs,
                hz=hz,
                max_jump_u=max_jump_u,
            )
            _write_end_record(stream, stats, stopped=stop["f"])
    finally:
        _cleanup_resources(drone, resources)

    print(f"[session] wrote {stats.samples} samples -> {out}", flush=True)
    print(
        f"[session] summary video_ok={stats.video_ok} video_none={stats.video_none} "
        f"loc_ok={stats.loc_ok} loc_fail={stats.loc_fail} "
        f"jump_rejects={stats.jump_rejects}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Passive flight session logger (no arm)")
    ap.add_argument("--ip", default="192.168.53.1")
    ap.add_argument("--controller", default="skycontroller3")
    ap.add_argument("--secs", type=float, default=900.0,
                    help="max duration; Ctrl-C ends early")
    ap.add_argument("--hz", type=float, default=5.0,
                    help="sample rate (localize is heavy; 5 Hz default)")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-localize", action="store_true",
                    help="telemetry+video only (skip model load)")
    args = ap.parse_args()

    if args.out:
        out = Path(args.out)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = (Path(pff.LOC_ROOT) / "outputs" / "flight_logs" /
               f"passive_session_{stamp}.jsonl")

    # ensure flight_control imports resolve when launched from elsewhere
    fc = Path(__file__).resolve().parent
    if str(fc) not in sys.path:
        sys.path.insert(0, str(fc))

    run_session(
        ip=args.ip,
        controller=args.controller,
        secs=args.secs,
        hz=args.hz,
        out=out,
        with_localize=not args.no_localize,
    )


if __name__ == "__main__":
    main()
