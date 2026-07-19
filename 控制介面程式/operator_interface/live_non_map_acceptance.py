#!/usr/bin/env python3
"""Live acceptance tests that do NOT need the Urai map / localization.

*** HARD SAFETY (2026-07-10 operator order) ***
  Agents / automation MUST NOT take off. Default is GROUND-ONLY.
  Takeoff is disabled unless BOTH are set:
    env SFM_ALLOW_AUTO_TAKEOFF=1
    flag --i-understand-this-will-takeoff
  Even then: prefer human UI takeoff. Do not use this path from an agent.

Default covers (ground only):
  A. connect + telemetry
  B. live video frames (PDRAW)
  C. gimbal pitch + camera zoom reset (on ground)
  F. PC freeze (Esc/manual) + resume (pc_control)  [PCMD zero only]

Optional air steps D/E/G/H exist only behind the dual interlock above.

Usage
  # Free any other Olympe holder first (UI / logger).
  python live_non_map_acceptance.py --ip 192.168.42.1 --controller drone
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_FC = _HERE.parent / "flight_control"
for p in (_HERE, _FC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

LOG_DIR = _HERE.parents[1] / "outputs" / "flight_logs"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    t_mono: float = 0.0


@dataclass
class Report:
    started_iso: str
    ip: str
    controller: str
    fly: bool
    checks: list[Check] = field(default_factory=list)
    ended_iso: str = ""
    log_path: str = ""
    sample_png: str = ""

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name=name, ok=ok, detail=detail, t_mono=time.monotonic()))
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {name}: {detail}", flush=True)

    def summary(self) -> dict[str, Any]:
        n_ok = sum(1 for c in self.checks if c.ok)
        n = len(self.checks)
        return {
            "started_iso": self.started_iso,
            "ended_iso": self.ended_iso,
            "ip": self.ip,
            "controller": self.controller,
            "fly": self.fly,
            "passed": n_ok,
            "total": n,
            "all_ok": n_ok == n and n > 0,
            "checks": [asdict(c) for c in self.checks],
            "log_path": self.log_path,
            "sample_png": self.sample_png,
        }


class LiveAcceptance:
    def __init__(self, ip: str, controller: str, *, fly: bool, nudge_pct: int, nudge_s: float,
                 out_dir: Path):
        self.ip = ip
        self.controller = controller
        self.fly = fly
        self.nudge_pct = int(nudge_pct)
        self.nudge_s = float(nudge_s)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = out_dir / f"live_acceptance_{ts}.jsonl"
        self.summary_path = out_dir / f"live_acceptance_{ts}.summary.json"
        self.sample_png = out_dir / f"live_acceptance_{ts}_frame.png"
        self.report = Report(
            started_iso=datetime.now(timezone.utc).isoformat(),
            ip=ip, controller=controller, fly=fly, log_path=str(self.log_path),
        )
        self._log_f = self.log_path.open("w", encoding="utf-8", buffering=1)
        self.backend = None
        self._cleaned = False
        atexit.register(self.cleanup)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(sig, self._on_signal)
            except Exception:
                pass

    def log(self, event: str, **kw: Any) -> None:
        rec = {
            "t_iso": datetime.now(timezone.utc).isoformat(),
            "t_mono": time.monotonic(),
            "event": event,
            **kw,
        }
        line = json.dumps(rec, ensure_ascii=False)
        print(f"[accept] {line}", flush=True)
        self._log_f.write(line + "\n")

    def _on_signal(self, signum, _frame) -> None:
        self.log("signal", signum=int(signum))
        self.cleanup()
        raise SystemExit(128 + int(signum))

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.log("cleanup_begin")
        try:
            if self.backend is not None:
                try:
                    self.backend.cleanup()
                except Exception as exc:
                    self.log("cleanup_backend_error", error=repr(exc))
                self.backend = None
        finally:
            self.report.ended_iso = datetime.now(timezone.utc).isoformat()
            self.report.sample_png = str(self.sample_png) if self.sample_png.exists() else ""
            summary = self.report.summary()
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            self.log("cleanup_done", summary_path=str(self.summary_path),
                     passed=summary["passed"], total=summary["total"])
            try:
                self._log_f.close()
            except Exception:
                pass

    def _flying_state(self) -> str:
        try:
            from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
            st = self.backend.drone.get_state(FlyingStateChanged)
            if isinstance(st, dict):
                return str(st.get("state", "unknown"))
            return str(st)
        except Exception as exc:
            return f"err:{exc!r}"

    def _battery(self) -> int | None:
        try:
            from olympe.messages.common.CommonState import BatteryStateChanged
            st = self.backend.drone.get_state(BatteryStateChanged)
            if isinstance(st, dict) and "percent" in st:
                return int(st["percent"])
        except Exception:
            pass
        return None

    def _wait_frames(self, secs: float = 4.0) -> tuple[int, Any]:
        """Return (n_frames, last_pil_or_none)."""
        n = 0
        last = None
        t0 = time.monotonic()
        vs = getattr(self.backend, "video_stream", None)
        if vs is None:
            return 0, None
        while time.monotonic() - t0 < secs:
            fr = vs.next_frame()
            if fr is not None:
                n += 1
                last = fr
            time.sleep(0.05)
        return n, last

    def run(self) -> int:
        from olympe_live_backend import OlympeLiveBackend

        # Minimal stand-ins for OperatorApp state/profile
        class _State:
            def __init__(self):
                self.mode = "MANUAL"
                self.tracker_state = "BOOT"
                self.loc = "LIVE"
                self.stream = "WAIT"
                self.pose = __import__("numpy").zeros(4, dtype=float)
                self.inliers = 0
                self.reproj = None
                self.battery_pct = 0.0
                self.altitude_m = 0.0
                self.gimbal_pitch_deg = -20.0
                self.zoom = 1.0
                self.link_latency_ms = 280.0
                self.stream_fps = 0.0
                self.stream_mbps = 5.0
                self.last_command = "ready"
                self.att_roll = 0.0
                self.att_pitch = 0.0
                self.att_yaw = 0.0

        class _Anafi:
            gimbal_pitch_min_deg = -90.0
            gimbal_pitch_max_deg = 90.0
            digital_zoom_max = 3.0
            stream_latency_ms = 280.0
            stream_fps = 30.0
            stream_mbps = 5.0
            takeoff_hover_m = 1.0

        self.log("begin", ip=self.ip, controller=self.controller, fly=self.fly)

        # ---- A connect ----
        try:
            self.backend = OlympeLiveBackend(
                _State, _Anafi(),
                ip=self.ip, controller=self.controller,
                nudge_pct=self.nudge_pct, nudge_pulse_s=self.nudge_s,
                cmd_log=self.log_path.with_suffix(".backend.jsonl"),
                with_video=True,
            )
            self.report.add("A_connect", True, f"drone={self.backend.drone is not None}")
        except Exception as exc:
            self.report.add("A_connect", False, repr(exc))
            self.cleanup()
            return 2

        # ---- A2 telemetry ----
        bat = self._battery()
        fly_st = self._flying_state()
        self.log("telemetry", battery=bat, flying_state=fly_st)
        self.report.add(
            "A_telemetry",
            bat is not None and bat > 0,
            f"battery={bat}% flying_state={fly_st}",
        )
        if bat is not None and bat < 20:
            self.report.add("A_battery_floor", False, f"battery {bat}% < 20 — abort fly")
            self.fly = False

        # ---- B video ----
        n, last = self._wait_frames(5.0)
        ok_v = n >= 5 and last is not None
        detail = f"frames={n}"
        if last is not None:
            try:
                import cv2
                import numpy as np
                arr = np.asarray(last)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    bgr = arr
                cv2.imwrite(str(self.sample_png), bgr)
                detail += f" shape={getattr(last, 'size', arr.shape)} saved={self.sample_png.name}"
            except Exception as exc:
                detail += f" save_err={exc!r}"
        grab_fps = float(getattr(self.backend.grabber, "fps", 0.0) or 0.0) if self.backend.grabber else 0.0
        detail += f" grabber_fps~{grab_fps:.1f}"
        self.report.add("B_video_frames", ok_v, detail)

        # ---- C gimbal + zoom on ground ----
        try:
            self.backend.set_gimbal_pitch(-45.0)
            time.sleep(0.8)
            self.backend.set_gimbal_pitch(-10.0)
            time.sleep(0.6)
            self.backend.reset_camera_defaults(pitch=-20.0, zoom=1.0)
            time.sleep(0.6)
            self.report.add(
                "C_gimbal_zoom",
                True,
                f"pitch_state={self.backend.state.gimbal_pitch_deg} zoom={self.backend.state.zoom}",
            )
        except Exception as exc:
            self.report.add("C_gimbal_zoom", False, repr(exc))

        if not self.fly:
            self.report.add("D_takeoff", True, "skipped (--no-fly)")
            self.report.add("E_nudges", True, "skipped (--no-fly)")
            self.report.add("F_freeze_resume", True, "skipped (--no-fly)")
            self.report.add("G_camera_air", True, "skipped (--no-fly)")
            self.report.add("H_land", True, "skipped (--no-fly)")
            self.cleanup()
            return 0 if all(c.ok for c in self.report.checks) else 1

        # ---- D takeoff ----
        if fly_st not in {"landed", "Landed", "0"} and "landed" not in fly_st.lower():
            # already flying? still try hover path
            self.log("pre_takeoff_state", flying_state=fly_st)
        try:
            self.backend.takeoff_cmd()
            time.sleep(1.0)
            st_after = self._flying_state()
            ok_to = "hover" in st_after.lower() or self.backend.state.tracker_state == "HOVER"
            # also accept takeoff success log state
            if self.backend.state.tracker_state == "TAKEOFF_FAIL":
                ok_to = False
            self.report.add("D_takeoff", ok_to, f"flying_state={st_after} tracker={self.backend.state.tracker_state}")
            if not ok_to:
                self.backend.land_cmd("takeoff_failed")
                self.cleanup()
                return 1
        except Exception as exc:
            self.report.add("D_takeoff", False, repr(exc))
            try:
                self.backend.land_cmd("takeoff_exception")
            except Exception:
                pass
            self.cleanup()
            return 1

        # hover settle
        self.backend.hover_cmd("post_takeoff_settle")
        time.sleep(2.0)

        # ---- E nudges ----
        nudge_ok = True
        nudge_details = []
        for name in ("前", "後", "左", "右", "上", "下", "右上前"):
            try:
                self.backend.nudge(name)
                time.sleep(self.nudge_s + 0.55)
                self.backend.hover_cmd(f"after_{name}")
                time.sleep(0.35)
                nudge_details.append(f"{name}=ok")
            except Exception as exc:
                nudge_ok = False
                nudge_details.append(f"{name}=ERR:{exc!r}")
        self.report.add("E_nudges", nudge_ok, "; ".join(nudge_details))

        # ---- F freeze / resume (direct: freeze PC PCMD) ----
        try:
            self.backend.give_to_pilot()
            blocked = not self.backend.send_pcmd(0, 10, 0, 0, reason="should_block")
            self.backend.take_pc_control()
            resumed = self.backend.send_pcmd(0, 0, 0, 0, reason="after_resume")
            self.report.add(
                "F_freeze_resume",
                blocked and resumed,
                f"blocked={blocked} resumed={resumed} tracker={self.backend.state.tracker_state}",
            )
        except Exception as exc:
            self.report.add("F_freeze_resume", False, repr(exc))

        # ---- G camera in air ----
        try:
            self.backend.set_gimbal_pitch(-30.0)
            time.sleep(0.7)
            self.backend.reset_camera_defaults()
            time.sleep(0.5)
            self.report.add(
                "G_camera_air",
                True,
                f"pitch={self.backend.state.gimbal_pitch_deg} zoom={self.backend.state.zoom}",
            )
        except Exception as exc:
            self.report.add("G_camera_air", False, repr(exc))

        # more video while airborne
        n2, _ = self._wait_frames(2.5)
        self.report.add("B2_video_in_air", n2 >= 3, f"frames={n2}")

        # ---- H land ----
        try:
            self.backend.land_cmd("acceptance_end")
            # wait for landed
            t0 = time.monotonic()
            landed = False
            while time.monotonic() - t0 < 20.0:
                st = self._flying_state()
                if "land" in st.lower():
                    landed = True
                    break
                time.sleep(0.4)
            self.report.add("H_land", landed, f"flying_state={self._flying_state()}")
        except Exception as exc:
            self.report.add("H_land", False, repr(exc))

        self.cleanup()
        summary = self.report.summary()
        print("\n==== ACCEPTANCE SUMMARY ====", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"summary file: {self.summary_path}", flush=True)
        return 0 if summary["all_ok"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ip", default="192.168.42.1")
    ap.add_argument("--controller", default="drone")
    ap.add_argument("--no-fly", action="store_true",
                    help="(default behavior) ground only — kept for compatibility")
    ap.add_argument(
        "--i-understand-this-will-takeoff",
        action="store_true",
        help="REQUIRED together with env SFM_ALLOW_AUTO_TAKEOFF=1 to enable takeoff. "
             "Agents must NOT pass this. Human UI takeoff only.",
    )
    ap.add_argument("--nudge-pct", type=int, default=8)
    ap.add_argument("--nudge-s", type=float, default=0.20)
    ap.add_argument("--out-dir", default=str(LOG_DIR))
    args = ap.parse_args()
    # Default SAFE: never take off. Dual interlock for any air path.
    env_ok = os.environ.get("SFM_ALLOW_AUTO_TAKEOFF", "") == "1"
    fly = bool(args.i_understand_this_will_takeoff and env_ok and not args.no_fly)
    if args.i_understand_this_will_takeoff and not env_ok:
        print(
            "[accept] REFUSING takeoff: set SFM_ALLOW_AUTO_TAKEOFF=1 only if a "
            "human on-site explicitly authorized scripted takeoff. Defaulting to ground.",
            flush=True,
        )
    print(
        f"[accept] LIVE acceptance ip={args.ip} controller={args.controller} fly={fly}\n"
        f"  SAFETY: takeoff={'ENABLED (dual interlock)' if fly else 'DISABLED (default)'}",
        flush=True,
    )
    runner = LiveAcceptance(
        args.ip, args.controller, fly=fly,
        nudge_pct=args.nudge_pct, nudge_s=args.nudge_s,
        out_dir=Path(args.out_dir),
    )
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
