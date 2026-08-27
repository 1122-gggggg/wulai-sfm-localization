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
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

LOG_DIR = _HERE.parents[1] / "outputs" / "flight_logs"


@dataclass
class Check:
    name: str
    ok: bool | None            # True=pass, False=fail, None=not run
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

    def skip(self, name: str, detail: str = "") -> None:
        """Record a check that did not run. A skipped check is NOT a pass.

        Recording skips as True made a --no-fly run report every air-phase check
        as PASS, which reads as "the aircraft was verified in flight".
        """
        self.checks.append(
            Check(name=name, ok=None, detail=detail, t_mono=time.monotonic()))
        print(f"[SKIP] {name}: {detail}", flush=True)

    def summary(self) -> dict[str, Any]:
        n_ok = sum(1 for c in self.checks if c.ok is True)
        n_failed = sum(1 for c in self.checks if c.ok is False)
        n_skipped = sum(1 for c in self.checks if c.ok is None)
        n = len(self.checks)
        return {
            "started_iso": self.started_iso,
            "ended_iso": self.ended_iso,
            "ip": self.ip,
            "controller": self.controller,
            "fly": self.fly,
            "passed": n_ok,
            "failed": n_failed,
            "skipped": n_skipped,
            "total": n,
            "all_ok": n_failed == 0 and n_ok > 0,
            "checks": [asdict(c) for c in self.checks],
            "log_path": self.log_path,
            "sample_png": self.sample_png,
        }


class _AcceptanceState:
    def __init__(self):
        import numpy as np

        self.mode = "MANUAL"
        self.tracker_state = "BOOT"
        self.loc = "LIVE"
        self.stream = "WAIT"
        self.pose = np.zeros(4, dtype=float)
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


class _AcceptanceAnafi:
    gimbal_pitch_min_deg = -90.0
    gimbal_pitch_max_deg = 90.0
    digital_zoom_max = 3.0
    stream_latency_ms = 280.0
    stream_fps = 30.0
    stream_mbps = 5.0
    takeoff_hover_m = 1.0


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

    def _connect_backend(self, backend_type) -> bool:
        try:
            self.backend = backend_type(
                _AcceptanceState,
                _AcceptanceAnafi(),
                ip=self.ip,
                controller=self.controller,
                nudge_pct=self.nudge_pct,
                nudge_pulse_s=self.nudge_s,
                cmd_log=self.log_path.with_suffix(".backend.jsonl"),
                with_video=True,
            )
            self.report.add(
                "A_connect",
                True,
                f"drone={self.backend.drone is not None}",
            )
            return True
        except Exception as exc:
            self.report.add("A_connect", False, repr(exc))
            self.cleanup()
            return False

    def _check_telemetry(self) -> str:
        battery = self._battery()
        flying_state = self._flying_state()
        self.log("telemetry", battery=battery, flying_state=flying_state)
        self.report.add(
            "A_telemetry",
            battery is not None and battery > 0,
            f"battery={battery}% flying_state={flying_state}",
        )
        if battery is not None and battery < 20:
            self.report.add(
                "A_battery_floor",
                False,
                f"battery {battery}% < 20 — abort fly",
            )
            self.fly = False
        return flying_state

    def _save_video_sample(self, frame) -> str:
        if frame is None:
            return ""
        try:
            import cv2
            import numpy as np

            array = np.asarray(frame)
            image = (
                cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                if array.ndim == 3 and array.shape[2] == 3
                else array
            )
            cv2.imwrite(str(self.sample_png), image)
            shape = getattr(frame, "size", array.shape)
            return f" shape={shape} saved={self.sample_png.name}"
        except Exception as exc:
            return f" save_err={exc!r}"

    def _check_video(self) -> None:
        frame_count, last_frame = self._wait_frames(5.0)
        detail = f"frames={frame_count}" + self._save_video_sample(last_frame)
        grabber = self.backend.grabber
        grabber_fps = float(getattr(grabber, "fps", 0.0) or 0.0) if grabber else 0.0
        detail += f" grabber_fps~{grabber_fps:.1f}"
        self.report.add(
            "B_video_frames",
            frame_count >= 5 and last_frame is not None,
            detail,
        )

    def _check_ground_camera(self) -> None:
        try:
            pitch_first_ok = bool(self.backend.set_gimbal_pitch(-45.0))
            time.sleep(0.8)
            pitch_second_ok = bool(self.backend.set_gimbal_pitch(-10.0))
            time.sleep(0.6)
            reset_ok = bool(
                self.backend.reset_camera_defaults(pitch=-20.0, zoom=1.0)
            )
            time.sleep(0.6)
            pitch_state = self.backend.state.gimbal_pitch_deg
            zoom_state = self.backend.state.zoom
            gimbal_ok = isinstance(pitch_state, (int, float)) and abs(
                float(pitch_state) + 20.0
            ) <= 3.0
            zoom_ok = isinstance(zoom_state, (int, float)) and abs(
                float(zoom_state) - 1.0
            ) <= 1e-6
            commands_ok = pitch_first_ok and pitch_second_ok and reset_ok
            self.report.add(
                "C_gimbal_zoom",
                bool(commands_ok and gimbal_ok and zoom_ok),
                f"commands={commands_ok} pitch_state={pitch_state} "
                f"(want -20+/-3) zoom={zoom_state} (want 1.0)",
            )
        except Exception as exc:
            self.report.add("C_gimbal_zoom", False, repr(exc))

    def _finish_ground_only(self) -> int:
        for name in (
            "D_takeoff",
            "E_nudges",
            "F_freeze_resume",
            "G_camera_air",
            "H_land",
        ):
            self.report.skip(name, "not run (--no-fly)")
        self.cleanup()
        return 1 if any(check.ok is False for check in self.report.checks) else 0

    def _takeoff(self, control_action, control_request, control_result, flying_state: str) -> bool:
        if flying_state not in {"landed", "Landed", "0"} and "landed" not in flying_state.lower():
            self.log("pre_takeoff_state", flying_state=flying_state)
        try:
            result = self.backend.command(
                control_request.create(control_action.TAKEOFF, human_origin=True)
            )
            time.sleep(1.0)
            state_after = self._flying_state()
            accepted = (
                bool(result.accepted)
                if isinstance(result, control_result)
                else bool(result)
            )
            succeeded = accepted and (
                "hover" in state_after.lower()
                or self.backend.state.tracker_state == "HOVER"
            )
            if self.backend.state.tracker_state == "TAKEOFF_FAIL":
                succeeded = False
            self.report.add(
                "D_takeoff",
                succeeded,
                f"flying_state={state_after} tracker={self.backend.state.tracker_state}",
            )
            if not succeeded:
                self.backend.land_cmd("takeoff_failed")
                self.cleanup()
            return succeeded
        except Exception as exc:
            self.report.add("D_takeoff", False, repr(exc))
            self._land_after_takeoff_exception()
            self.cleanup()
            return False

    def _land_after_takeoff_exception(self) -> None:
        try:
            self.backend.land_cmd("takeoff_exception")
        except Exception as exc:
            self.report.add("D_takeoff_land", False, repr(exc))
            print(f"[acceptance] emergency land FAILED: {exc!r}", flush=True)

    def _check_nudges(self) -> None:
        succeeded = True
        details = []
        for name in ("前", "後", "左", "右", "上", "下", "右上前"):
            try:
                accepted = bool(self.backend.nudge_begin(name))
                time.sleep(self.nudge_s + 0.55)
                self.backend.nudge_end(name)
                self.backend.hover_cmd(f"after_{name}")
                time.sleep(0.35)
                succeeded = succeeded and accepted
                details.append(f"{name}={'ok' if accepted else 'REFUSED'}")
            except Exception as exc:
                succeeded = False
                details.append(f"{name}=ERR:{exc!r}")
        self.report.add("E_nudges", succeeded, "; ".join(details))

    def _check_freeze_resume(self) -> None:
        try:
            self.backend.give_to_pilot()
            blocked = not self.backend.send_pcmd(
                0,
                10,
                0,
                0,
                reason="should_block",
            )
            self.backend.take_pc_control()
            resumed = self.backend.send_pcmd(
                0,
                0,
                0,
                0,
                reason="after_resume",
            )
            self.report.add(
                "F_freeze_resume",
                blocked and resumed,
                f"blocked={blocked} resumed={resumed} "
                f"tracker={self.backend.state.tracker_state}",
            )
        except Exception as exc:
            self.report.add("F_freeze_resume", False, repr(exc))

    def _check_air_camera(self) -> None:
        try:
            pitch_ok = bool(self.backend.set_gimbal_pitch(-30.0))
            time.sleep(0.7)
            reset_ok = bool(self.backend.reset_camera_defaults())
            time.sleep(0.5)
            self.report.add(
                "G_camera_air",
                bool(pitch_ok and reset_ok),
                f"commands={pitch_ok and reset_ok} "
                f"pitch={self.backend.state.gimbal_pitch_deg} "
                f"zoom={self.backend.state.zoom}",
            )
        except Exception as exc:
            self.report.add("G_camera_air", False, repr(exc))

    def _check_air_video(self) -> None:
        frame_count, _frame = self._wait_frames(2.5)
        self.report.add("B2_video_in_air", frame_count >= 3, f"frames={frame_count}")

    def _check_landing(self) -> None:
        try:
            self.backend.land_cmd("acceptance_end")
            started_at = time.monotonic()
            landed = False
            while time.monotonic() - started_at < 20.0:
                if "land" in self._flying_state().lower():
                    landed = True
                    break
                time.sleep(0.4)
            self.report.add(
                "H_land",
                landed,
                f"flying_state={self._flying_state()}",
            )
        except Exception as exc:
            self.report.add("H_land", False, repr(exc))

    def _finish_report(self) -> int:
        self.cleanup()
        summary = self.report.summary()
        print("\n==== ACCEPTANCE SUMMARY ====", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"summary file: {self.summary_path}", flush=True)
        return 0 if summary["all_ok"] else 1

    def run(self) -> int:
        from backend_contract import ControlAction, ControlRequest, ControlResult
        from olympe_live_backend import OlympeLiveBackend

        self.log("begin", ip=self.ip, controller=self.controller, fly=self.fly)
        if not self._connect_backend(OlympeLiveBackend):
            return 2
        flying_state = self._check_telemetry()
        self._check_video()
        self._check_ground_camera()
        if not self.fly:
            return self._finish_ground_only()
        if not self._takeoff(
            ControlAction,
            ControlRequest,
            ControlResult,
            flying_state,
        ):
            return 1
        self.backend.hover_cmd("post_takeoff_settle")
        time.sleep(2.0)
        self._check_nudges()
        self._check_freeze_resume()
        self._check_air_camera()
        self._check_air_video()
        self._check_landing()
        return self._finish_report()


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
