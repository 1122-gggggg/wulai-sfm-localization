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


def run_session(*, ip: str, controller: str, secs: float, hz: float,
                out: Path, with_localize: bool) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    messages = tel._build_message_table()

    stop = {"f": False}

    def _sig(*_a):
        stop["f"] = True
        print("[session] stop requested (signal)", flush=True)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[session] connecting ip={ip} controller={controller}", flush=True)
    print("[session] PASSIVE ONLY — no TakeOff/PCMD/Landing/Emergency/gimbal cmd", flush=True)
    drone = ofs.connect(ip, controller=controller)

    grabber = None
    loc = None
    waypoints = None
    cum = None
    path_len = None

    try:
        grabber = ofs.OlympePdrawGrabber(
            drone, resize=(1280, 720), stale_s=0.8).start()
        print("[session] video grabber started", flush=True)

        if with_localize:
            print("[session] loading localizer models (stay on ground until ready)...",
                  flush=True)
            loc = pff.build_localizer(grabber)
            loc.ensure_models()
            try:
                waypoints = rpf.load_waypoints(pff.PATH_JSON)
                poles = rpf.load_poles(pff.POLES_JSON)
                ctrl = rpf.RouteAutoController(waypoints, poles)
                cum = ctrl.cum
                path_len = float(ctrl.path_len)
                print(f"[session] route loaded: {len(waypoints)} wp, "
                      f"len={path_len:.2f}u  poles={len(poles)}", flush=True)
            except Exception as exc:
                print(f"[session] route load failed (loc continues): {exc}", flush=True)
            print("[session] localizer ready — pilot may take off with sticks", flush=True)
    except Exception as exc:
        print(f"[session] video/localizer setup failed: {exc}", flush=True)
        traceback.print_exc()
        # telemetry-only fallback
        if grabber is not None:
            try:
                grabber.stop()
            except Exception:
                pass
            grabber = None
        loc = None

    period = 1.0 / max(1e-3, hz)
    t_end = time.monotonic() + max(0.1, secs)
    n = 0
    video_ok = video_none = 0
    loc_ok = loc_fail = loc_skip = 0
    jump_rejects = 0
    last_good_xyz = None
    max_jump_u = float(getattr(pff, "MAX_POSE_JUMP_U", 1.5)
                       if hasattr(pff, "MAX_POSE_JUMP_U") else
                       __import__("os").environ.get("SFM_MAX_POSE_JUMP_U", "1.5"))

    # use same threshold as flight if available
    try:
        max_jump_u = pff._env_float("SFM_MAX_POSE_JUMP_U", 1.5, minimum=0.1, maximum=50.0)
    except Exception:
        max_jump_u = 1.5

    try:
        with out.open("w", encoding="utf-8") as f:
            meta = {
                "event": "start",
                "ip": ip,
                "controller": controller,
                "secs": secs,
                "hz": hz,
                "with_localize": bool(with_localize and loc is not None),
                "bundle": str(pff.XBUN),
                "path_json": str(pff.PATH_JSON),
                "channel_labels": [lab for lab, _ in messages],
                "safety": "passive_no_arm",
                "max_pose_jump_u": max_jump_u,
                "t_iso": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            f.flush()

            while (not stop["f"]) and time.monotonic() < t_end:
                t0 = time.monotonic()
                row = tel.sample_once(drone, messages)

                # video
                frame = None
                stamp = None
                if grabber is not None:
                    try:
                        fr = grabber()
                    except Exception:
                        fr = None
                    if fr is None:
                        video_none += 1
                        row["video.frame"] = None
                        row["video.healthy"] = bool(grabber.is_healthy())
                    else:
                        video_ok += 1
                        frame, stamp = fr if isinstance(fr, tuple) else (fr, None)
                        h, w = (frame.shape[:2] if frame is not None else (None, None))
                        row["video.frame"] = {
                            "w": w, "h": h,
                            "stamp_mono": stamp,
                            "age_s": (time.monotonic() - stamp) if stamp else None,
                        }
                        row["video.healthy"] = bool(grabber.is_healthy())
                        row["video.fps"] = float(getattr(grabber, "fps", 0.0) or 0.0)
                else:
                    row["video.frame"] = None

                # localization (uses grabber as frame_source; get_pose pulls frame)
                if loc is not None:
                    try:
                        pose = loc.get_pose()
                        info = dict(getattr(loc, "last_info", {}) or {})
                        finite = _finite_pose(pose)
                        jump = False
                        if finite and last_good_xyz is not None:
                            dx = pose.x - last_good_xyz[0]
                            dy = pose.y - last_good_xyz[1]
                            dz = pose.z - last_good_xyz[2]
                            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                            if dist > max_jump_u:
                                jump = True
                                jump_rejects += 1
                        if finite and not jump:
                            last_good_xyz = (float(pose.x), float(pose.y), float(pose.z))
                            loc_ok += 1
                        else:
                            loc_fail += 1

                        path_err = None
                        progress = None
                        if finite and waypoints is not None and cum is not None:
                            try:
                                d, _nseg, _seg, s = rpf.project_to_path(
                                    [pose.x, pose.y, pose.z], waypoints, cum)
                                path_err = float(d)
                                progress = float(s / path_len) if path_len else None
                            except Exception:
                                pass

                        row["loc"] = {
                            "ok": bool(finite and not jump),
                            "finite": finite,
                            "jump_reject": jump,
                            "x": float(pose.x) if finite else None,
                            "y": float(pose.y) if finite else None,
                            "z": float(pose.z) if finite else None,
                            "yaw": float(getattr(pose, "yaw", float("nan")))
                                   if finite else None,
                            "stamp": float(getattr(pose, "stamp", 0.0) or 0.0)
                                     if finite else None,
                            "mode": info.get("mode") or info.get("next_mode"),
                            "inliers": int(info.get("inliers", 0) or 0),
                            "weak": bool(info.get("weak", False)),
                            "reproj": info.get("reproj") or info.get("reproj_rms"),
                            "path_error_u": path_err,
                            "progress": progress,
                        }
                    except Exception as exc:
                        loc_fail += 1
                        row["loc"] = {"ok": False, "error": repr(exc)}
                else:
                    loc_skip += 1
                    row["loc"] = None

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
                if n % max(1, int(hz)) == 0:
                    f.flush()
                    att = row.get("piloting.attitude")
                    alt = row.get("piloting.altitude")
                    fly = row.get("piloting.flying_state")
                    bat = row.get("common.battery") or row.get("battery.capacity")
                    loc_s = row.get("loc") or {}
                    print(
                        f"[session] n={n} fly={fly} alt={alt} bat={bat} "
                        f"vid_ok={video_ok} loc_ok={loc_ok}/{loc_ok + loc_fail} "
                        f"mode={loc_s.get('mode')} inl={loc_s.get('inliers')} "
                        f"pos=({loc_s.get('x')},{loc_s.get('y')},{loc_s.get('z')}) "
                        f"path_err={loc_s.get('path_error_u')}",
                        flush=True,
                    )

                dt = time.monotonic() - t0
                time.sleep(max(0.0, period - dt))

            footer = {
                "event": "end",
                "samples": n,
                "video_frames_ok": video_ok,
                "video_frames_none": video_none,
                "loc_ok": loc_ok,
                "loc_fail": loc_fail,
                "loc_skip": loc_skip,
                "jump_rejects": jump_rejects,
                "stopped": stop["f"],
                "t_iso": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(footer, ensure_ascii=False) + "\n")
    finally:
        if grabber is not None:
            try:
                grabber.stop()
            except Exception:
                pass
        try:
            drone.disconnect()
        except Exception:
            pass

    print(f"[session] wrote {n} samples -> {out}", flush=True)
    print(
        f"[session] summary video_ok={video_ok} video_none={video_none} "
        f"loc_ok={loc_ok} loc_fail={loc_fail} jump_rejects={jump_rejects}",
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
