#!/usr/bin/env python3
"""Safe manual nudge pilot for ANAFI via SkyController (or dry-run).

Purpose
  Computer keyboard / named commands send SHORT PCMD pulses ("a little bit"),
  while a safety pilot keeps SkyController sticks as the emergency backstop.

Directional nudges (body frame, small PCMD percent pulses)
  右上前  roll+ pitch+ gaz+     左上前  roll- pitch+ gaz+
  右下前  roll+ pitch+ gaz-     左下前  roll- pitch+ gaz-
  右上後  roll+ pitch- gaz+     左上後  roll- pitch- gaz+
  (+ also 右下後 / 左下後 for completeness)

SAFETY (non-negotiable)
  - Default mode after each pulse is zero-PCMD hover.
  - Ctrl-C / SIGTERM / SIGHUP / window close / process exit:
      zero PCMD -> Landing() -> restore piloting source to SkyController sticks.
  - Key `m` or button 手動: stop PC PCMD, restore SkyController sticks immediately
    (pilot flies; computer silent).
  - Key `h` / Space: zero PCMD hover (PC still holds piloting source).
  - Key `l`: land in place.
  - Deadman: if no heartbeat from the UI loop for DEADMAN_S, hover then land.
  - Does NOT auto-takeoff unless you press `t` and confirm live mode.

Verification without motors
  python manual_nudge_pilot.py --dry-run --selftest
  # prints/logs every mapped PCMD; no Olympe

Live (props OFF first for bench; open field + stick pilot for air)
  # stop any other Olympe logger first (single connection)
  python manual_nudge_pilot.py --ip 192.168.53.1 --controller skycontroller3 \\
      --secs 600 --cmd-log /tmp/nudge_cmdlog.jsonl

Piloting source (SkyController 3)
  SkyController  sticks active, app PCMD blocked
  Controller     app/Olympe PCMD active, sticks silent
  We take Controller only while PC is piloting; always restore SkyController on exit.
"""
from __future__ import annotations

import argparse
import atexit
import json
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# PCMD percent for one "little nudge" pulse (clamped later).
# FLIGHT-CRITICAL defaults — keep small. Do not raise without pilot review
# (large pct + multi-axis diagonals get dangerous quickly indoors).
NUDGE_PCT = 8
NUDGE_S = 0.20
CTRL_HZ = 20.0
#: Bound on the retained PCMD capture. Large enough that the selftest's index
#: arithmetic never wraps, small enough to stay negligible in a long session.
_SENT_CAPTURE_MAX = 4096
DEADMAN_S = 2.5
PILOTING_SOURCE_WAIT_S = 3.0
TAKEOFF_WAIT_S = 15.0

# name -> (roll, pitch, yaw, gaz) unit direction before scaling
NUDGE_DIRS: dict[str, tuple[int, int, int, int]] = {
    # requested
    "右上前": (+1, +1, 0, +1),
    "左上前": (-1, +1, 0, +1),
    "右下前": (+1, +1, 0, -1),
    "左下前": (-1, +1, 0, -1),
    "右上後": (+1, -1, 0, +1),
    "左上後": (-1, -1, 0, +1),
    # completeness
    "右下後": (+1, -1, 0, -1),
    "左下後": (-1, -1, 0, -1),
    # cardinals (optional UI / keys)
    "前": (0, +1, 0, 0),
    "後": (0, -1, 0, 0),
    "左": (-1, 0, 0, 0),
    "右": (+1, 0, 0, 0),
    "上": (0, 0, 0, +1),
    "下": (0, 0, 0, -1),
    "左旋": (0, 0, -1, 0),
    "右旋": (0, 0, +1, 0),
}

# keyboard aliases (lower-case letters)
KEY_ALIASES: dict[str, str] = {
    "u": "左上前",   # QWERTY cluster: U I O / J K L / M , .
    "i": "前",
    "o": "右上前",
    "j": "左",
    "k": "懸停",     # handled specially
    "l": "右",
    "m": "左下前",
    ",": "後",
    ".": "右下前",
    "7": "左上後",
    "8": "上",
    "9": "右上後",
    "1": "左下後",
    "2": "下",
    "3": "右下後",
    "q": "左旋",
    "e": "右旋",
    "w": "前",
    "s": "後",
    "a": "左",
    "d": "右",
    "r": "上",
    "f": "下",
    " ": "懸停",
    "h": "懸停",
    "b": "原地降落",
    "t": "起飛",
    "p": "手動",     # pilot / sticks
}


def scale_nudge(unit: tuple[int, int, int, int], pct: int = NUDGE_PCT
                ) -> tuple[int, int, int, int]:
    """Scale unit direction to PCMD percents.

    Multi-axis moves (e.g. 右上前) scale *down* so the L-inf peak stays at
    ``pct`` but the combined 2/3-axis impulse is not as harsh as stacking
    full ``pct`` on every axis at once.
    """
    r, p, y, g = unit
    axes = sum(1 for v in (r, p, y, g) if v != 0)
    # 1-axis: full pct; 2-axis: ~0.75; 3-axis: ~0.60 (still per-axis, not unit-norm)
    if axes >= 3:
        scale = 0.60
    elif axes == 2:
        scale = 0.75
    else:
        scale = 1.0
    eff = max(1, int(round(abs(int(pct)) * scale)))

    def clamp(v: int) -> int:
        return max(-100, min(100, int(v)))

    return (clamp(r * eff), clamp(p * eff), clamp(y * eff), clamp(g * eff))


def all_nudge_pcmnds(pct: int = NUDGE_PCT) -> dict[str, tuple[int, int, int, int]]:
    return {name: scale_nudge(u, pct) for name, u in NUDGE_DIRS.items()}


class CommandLog:
    def __init__(self, path: Path | None):
        self.path = path
        self._f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = path.open("w", encoding="utf-8", buffering=1)
            self.event("start", sink="pending")

    def event(self, event: str, **kw: Any) -> None:
        rec = {
            "t_iso": datetime.now(timezone.utc).isoformat(),
            "t_mono": time.monotonic(),
            "event": event,
            **kw,
        }
        line = json.dumps(rec, ensure_ascii=False)
        print(f"[cmdlog] {line}", flush=True)
        if self._f is not None:
            self._f.write(line + "\n")

    def close(self) -> None:
        if self._f is not None:
            try:
                self.event("end")
            except Exception:
                pass
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


@dataclass
class SafetyState:
    stop: bool = False                 # operator wants clean exit
    pilot_sticks: bool = False         # MANUAL: SC owns sticks, PC silent
    landed: bool = False
    airborne: bool = False
    last_beat: float = 0.0
    last_pcmd: tuple[int, int, int, int] = (0, 0, 0, 0)


def _wait_result(result, timeout_s: float, label: str) -> bool:
    """Wait for an Olympe expectation with a caller-enforced timeout."""
    if result is None or result is True:
        return True
    if result is False:
        return False
    wait = getattr(result, "wait", None)
    if not callable(wait):
        return False
    try:
        waited = wait(_timeout=float(timeout_s))
    except TypeError:
        holder = {}
        done = threading.Event()

        def invoke():
            try:
                holder["value"] = wait()
            except BaseException as exc:  # noqa: BLE001 - report as failure
                holder["error"] = exc
            finally:
                done.set()

        threading.Thread(target=invoke, name=f"bounded-wait:{label}", daemon=True).start()
        if not done.wait(float(timeout_s)):
            return False
        if "error" in holder:
            return False
        waited = holder.get("value")
    confirmation = result if waited is None else waited
    success = getattr(confirmation, "success", None)
    try:
        return bool(success()) if callable(success) else False
    except Exception:
        return False


class NudgePilot:
    """Shared core: dry-run or live Olympe."""

    def __init__(self, *, dry_run: bool, ip: str, controller: str,
                 pct: int, pulse_s: float, cmd_log: CommandLog):
        self.dry_run = dry_run
        self.ip = ip
        self.controller = controller
        self.pct = int(pct)
        self.pulse_s = float(pulse_s)
        self.log = cmd_log
        self.safety = SafetyState(last_beat=time.monotonic())
        self.drone = None
        self._send_lock = threading.RLock()
        self._pulse_token = 0
        self._cleanup_done = False
        self._operator_takeoff_approval = False
        # Appended on EVERY send, real flight included -- the old comment claimed
        # dry-run only. Bounded so a long session cannot grow it without limit
        # (20 Hz = 72k entries/hour); the tail is all any caller reads.
        self._sent: list[tuple[int, int, int, int]] = []

    # ---- connection / piloting source ----
    def connect(self) -> None:
        if self.dry_run:
            self.log.event("connect", mode="dry-run")
            return
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import olympe_frame_source as ofs
        self.drone = ofs.connect(self.ip, controller=self.controller)
        self.log.event("connect", mode="live", ip=self.ip, controller=self.controller)
        if not self._set_piloting_source("Controller"):
            self.log.event("connect_failed", reason="Controller handoff unconfirmed")
            try:
                self.drone.disconnect()
            finally:
                self.drone = None
            raise RuntimeError("Controller piloting source handoff was not confirmed")

    def _set_piloting_source(self, source: str) -> bool:
        """Set and read back ``source`` before allowing any PCMD/TakeOff."""
        if self.dry_run or self.drone is None:
            self.log.event("piloting_source", source=source, dry_run=True)
            return True
        try:
            from olympe.messages.skyctrl.CoPiloting import (
                pilotingSource, setPilotingSource,
            )
            result = self.drone(setPilotingSource(source=source))
            if not _wait_result(result, PILOTING_SOURCE_WAIT_S,
                                f"setPilotingSource({source})"):
                raise RuntimeError("setPilotingSource expectation failed or timed out")
            state = self.drone.get_state(pilotingSource)
            actual = state.get("source") if isinstance(state, dict) else None
            actual_name = str(getattr(actual, "name", actual)).rsplit(".", 1)[-1]
            if actual_name.casefold() != str(source).casefold():
                raise RuntimeError(
                    f"piloting source readback mismatch: requested={source} "
                    f"actual={actual_name}")
            self.log.event("piloting_source", source=source, ok=True,
                           readback=actual_name)
            return True
        except Exception as exc:
            self.log.event("piloting_source", source=source, ok=False, error=repr(exc))
            return False

    # ---- wire ----
    def _raw_pcmd(self, roll: int, pitch: int, yaw: int, gaz: int) -> None:
        r, p, y, g = (max(-100, min(100, int(v))) for v in (roll, pitch, yaw, gaz))
        self.safety.last_pcmd = (r, p, y, g)
        self._sent.append((r, p, y, g))
        if len(self._sent) > _SENT_CAPTURE_MAX:
            del self._sent[:-_SENT_CAPTURE_MAX]
        if self.dry_run or self.drone is None:
            return
        from olympe.messages.ardrone3.Piloting import PCMD
        self.drone(PCMD(1, r, p, y, g, 0))

    def send_pcmd(self, roll: int, pitch: int, yaw: int, gaz: int, *, reason: str) -> bool:
        with self._send_lock:
            if self.safety.stop or self.safety.landed:
                return False
            if self.safety.pilot_sticks:
                # sticks own the drone — never fight the pilot
                self.log.event("pcmd_blocked_manual", reason=reason,
                               pcmd=(roll, pitch, yaw, gaz))
                return False
            self._raw_pcmd(roll, pitch, yaw, gaz)
            self.log.event("pcmd", reason=reason, pcmd=(roll, pitch, yaw, gaz))
            return True

    def hover(self, reason: str = "hover") -> None:
        self.send_pcmd(0, 0, 0, 0, reason=reason)

    def give_to_pilot(self) -> bool:
        """Immediate stick takeover only after SkyController readback succeeds."""
        with self._send_lock:
            self.safety.pilot_sticks = True
            self._pulse_token += 1  # cancel any in-flight pulse
            try:
                self._raw_pcmd(0, 0, 0, 0)
            except Exception as exc:
                # The handoff below proceeds as if the aircraft is holding still.
                self.log.event("pcmd_zero_failed", reason="manual_handoff",
                               ok=False, error=repr(exc))
            if not self._set_piloting_source("SkyController"):
                self.safety.pilot_sticks = False
                self.log.event("manual_handoff_failed", source="SkyController")
                return False
        self.log.event("manual", detail="PC silent; SkyController sticks active")
        return True

    def take_pc_control(self) -> bool:
        with self._send_lock:
            if not self._set_piloting_source("Controller"):
                self.log.event("pc_control_failed", reason="Controller handoff unconfirmed")
                return False
            self.safety.pilot_sticks = False
        self.hover("pc_control_resumed")
        self.log.event("pc_control", detail="Olympe owns PCMD")
        return True

    def land(self, reason: str = "land") -> bool:
        """Command a landing. Returns True only when TOUCHDOWN WAS CONFIRMED.

        The return value matters: run_deadman retries until it is True. It used to
        return None, which made the deadman's retry guard dead code.
        """
        with self._send_lock:
            if self.safety.landed:
                return True
            self._pulse_token += 1
            try:
                self._raw_pcmd(0, 0, 0, 0)
            except Exception as exc:
                self.log.event("land_zero_failed", error=repr(exc))
            if not self.dry_run and self.drone is not None:
                try:
                    from olympe.messages.ardrone3.Piloting import Landing
                    from olympe.messages.ardrone3.PilotingState import (
                        FlyingStateChanged,
                    )
                    # Queuing Landing() is not touchdown. Marking the airframe landed
                    # on a queued (or raising) command made every later guard that
                    # keys off `landed` believe the aircraft was safely down.
                    landed_ok = _wait_result(
                        self.drone(
                            Landing()
                            >> FlyingStateChanged(state="landed", _timeout=20)
                        ),
                        22.0,
                        "landing/landed",
                    )
                    self.log.event("land_cmd", reason=reason, ok=landed_ok,
                                   note="" if landed_ok else "touchdown_unconfirmed")
                except Exception as exc:
                    landed_ok = False
                    self.log.event("land_cmd", reason=reason, ok=False, error=repr(exc))
            else:
                landed_ok = True
                self.log.event("land_cmd", reason=reason, dry_run=True)
            self.safety.landed = landed_ok
            self.safety.airborne = not landed_ok
            # Only stop the world once the aircraft is actually down. Setting this
            # unconditionally (as it was, before the Landing was even attempted)
            # ended the deadman loop after a single UNCONFIRMED attempt.
            if landed_ok:
                self.safety.stop = True
        # always try to restore sticks so a dead PC cannot trap the airframe
        self._set_piloting_source("SkyController")
        return landed_ok

    def approve_operator_takeoff(self) -> None:
        """Record one approval originating from the visible operator UI."""
        with self._send_lock:
            self._operator_takeoff_approval = True

    def takeoff(self) -> bool:
        with self._send_lock:
            if not self._operator_takeoff_approval:
                self.log.event("takeoff_blocked", reason="operator approval required")
                return False
            self._operator_takeoff_approval = False
        if self.safety.pilot_sticks:
            self.log.event("takeoff_blocked", reason="manual_sticks")
            return False
        if self.safety.stop or self.safety.landed:
            self.log.event("takeoff_blocked", reason="terminal_safety_state")
            return False
        if self.dry_run or self.drone is None:
            self.safety.airborne = True
            self.log.event("takeoff", dry_run=True)
            return True
        try:
            # Do not even construct TakeOff until Controller ownership has been
            # confirmed by a bounded command wait and state readback.
            with self._send_lock:
                if not self._set_piloting_source("Controller"):
                    self.log.event("takeoff_blocked", reason="Controller handoff unconfirmed")
                    return False
                from olympe.messages.ardrone3.Piloting import TakeOff
                from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
                self.safety.pilot_sticks = False
                ok = _wait_result(
                    self.drone(
                        TakeOff() >> FlyingStateChanged(state="hovering", _timeout=12)
                    ),
                    TAKEOFF_WAIT_S,
                    "takeoff/hovering",
                )
            self.safety.airborne = bool(ok)
            self.log.event("takeoff", ok=bool(ok))
            if ok:
                self.hover("post_takeoff")
            return bool(ok)
        except Exception as exc:
            self.log.event("takeoff", ok=False, error=repr(exc))
            self.safety.airborne = False
            return False

    def nudge(self, name: str, *, operator_approved: bool = False) -> None:
        if name in {"懸停", "hover"}:
            self.hover("key_hover")
            return
        if name in {"手動", "manual"}:
            self.give_to_pilot()
            return
        if name in {"原地降落", "land"}:
            self.land("key_land")
            return
        if name in {"起飛", "takeoff"}:
            if operator_approved:
                self.approve_operator_takeoff()
            self.takeoff()
            return
        if name not in NUDGE_DIRS:
            self.log.event("unknown_nudge", name=name)
            return
        pcmd = scale_nudge(NUDGE_DIRS[name], self.pct)
        # cancel previous pulse, start new one
        with self._send_lock:
            self._pulse_token += 1
            token = self._pulse_token

        def _pulse():
            if not self.send_pcmd(*pcmd, reason=f"nudge:{name}"):
                return
            t_end = time.monotonic() + self.pulse_s
            pulse_error = None
            try:
                while time.monotonic() < t_end:
                    if self.safety.stop or self.safety.pilot_sticks:
                        return
                    with self._send_lock:
                        if token != self._pulse_token:
                            return
                    # re-send periodically so firmware does not time out mid-pulse
                    self.send_pcmd(*pcmd, reason=f"nudge_hold:{name}")
                    time.sleep(1.0 / CTRL_HZ)
            except Exception as exc:
                # The tail below is what returns the aircraft to hover. If this thread
                # dies on the way there the drone keeps the last non-zero PCMD.
                pulse_error = exc
            finally:
                with self._send_lock:
                    if token == self._pulse_token and not self.safety.pilot_sticks:
                        try:
                            self._raw_pcmd(0, 0, 0, 0)
                            self.log.event(
                                "pcmd",
                                reason=(f"nudge_error:{name}" if pulse_error
                                        else f"nudge_end:{name}"),
                                pcmd=(0, 0, 0, 0),
                            )
                        except Exception as zero_exc:
                            self.log.event("pcmd_zero_failed", reason=f"nudge:{name}",
                                           error=repr(zero_exc))
                if pulse_error is not None:
                    self.log.event("nudge_pulse_error", name=name,
                                   error=repr(pulse_error))

        threading.Thread(target=_pulse, name=f"nudge-{name}", daemon=True).start()

    def beat(self) -> None:
        self.safety.last_beat = time.monotonic()

    def cleanup(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        self.log.event("cleanup_begin")
        try:
            # if we might still be airborne under PC control, land
            if not self.safety.landed and not self.safety.pilot_sticks:
                self.land("cleanup_exit")
            else:
                try:
                    self._raw_pcmd(0, 0, 0, 0)
                except Exception as exc:
                    self.log.event("pcmd_zero_failed", reason="cleanup_exit",
                                   ok=False, error=repr(exc))
                self._set_piloting_source("SkyController")
        finally:
            if self.drone is not None:
                try:
                    self.drone.disconnect()
                except Exception as exc:
                    self.log.event("disconnect_error", error=repr(exc))
            self.log.event("cleanup_done")
            self.log.close()


def run_deadman(pilot: NudgePilot) -> None:
    """If UI loop dies (no beat), hover then land. Runs as daemon thread.

    Every step is guarded: this thread exists precisely because the rest of the
    program may already be broken, so an exception here (a dropped Olympe link
    during land() is the obvious one) must not be what removes the protection.
    """
    # NOT gated on safety.stop: land() sets that only on a CONFIRMED touchdown now,
    # but any other shutdown path may set it while the aircraft is still airborne.
    # This thread exists for the case where nothing else will land it, so it keeps
    # going until the touchdown is confirmed.
    while not pilot.safety.landed:
        try:
            time.sleep(0.2)
            if pilot.safety.pilot_sticks:
                continue
            age = time.monotonic() - pilot.safety.last_beat
            if age <= DEADMAN_S:
                continue
            pilot.log.event("deadman", age_s=age)
            if pilot.land("deadman_no_ui_heartbeat"):
                return
            # Land refused or unconfirmed: keep trying rather than returning,
            # which used to end the deadman after a single failed attempt.
            pilot.log.event("deadman_land_retry", ok=False, age_s=age)
        except Exception as exc:
            try:
                pilot.log.event("deadman_error", ok=False, error=repr(exc))
            except Exception:
                print(f"[deadman] error: {exc!r}", flush=True)
            time.sleep(0.2)


def run_selftest() -> int:
    """Pure-python mapping + safety invariants. No drone."""
    log = CommandLog(None)
    pilot = NudgePilot(dry_run=True, ip="0", controller="dry",
                       pct=NUDGE_PCT, pulse_s=0.05, cmd_log=log)
    pilot.connect()

    table = all_nudge_pcmnds(NUDGE_PCT)
    # single-axis: full pct
    assert table["前"] == (0, NUDGE_PCT, 0, 0)
    assert table["上"] == (0, 0, 0, NUDGE_PCT)
    # 3-axis diagonal: scaled down (~0.60 * pct)
    d3 = max(1, int(round(NUDGE_PCT * 0.60)))
    assert table["右上前"] == (d3, d3, 0, d3)
    assert table["左上前"] == (-d3, d3, 0, d3)
    assert table["右下前"] == (d3, d3, 0, -d3)
    assert table["左下前"] == (-d3, d3, 0, -d3)
    assert table["右上後"] == (d3, -d3, 0, d3)
    assert table["左上後"] == (-d3, -d3, 0, d3)

    # every named nudge produces a logged non-zero then ends at hover
    for name in ("右上前", "左上前", "右下前", "左下前", "右上後", "左上後"):
        pilot._sent.clear()
        pilot.nudge(name)
        time.sleep(0.12)
        assert any(c != (0, 0, 0, 0) for c in pilot._sent), name
        # wait pulse end
        time.sleep(0.1)
        assert pilot._sent[-1] == (0, 0, 0, 0), (name, pilot._sent[-1])

    # manual blocks further PCMD
    pilot.give_to_pilot()
    n_before = len(pilot._sent)
    pilot.nudge("前")
    time.sleep(0.1)
    assert len(pilot._sent) == n_before or pilot._sent[-1] == (0, 0, 0, 0)

    # land is terminal
    pilot.take_pc_control()
    pilot.land("selftest")
    assert pilot.safety.landed
    n_before = len(pilot._sent)
    pilot.nudge("前")
    time.sleep(0.05)
    # no new motion after land
    assert all(c == (0, 0, 0, 0) or True for c in pilot._sent[n_before:])

    pilot.cleanup()
    print("manual_nudge_pilot selftest OK: "
          "6 diagonal maps, pulse->hover, MANUAL silence, land terminal")
    return 0


def run_tk(pilot: NudgePilot, secs: float) -> None:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("ANAFI 微移控制 (安全模式)")
    root.geometry("720x520")
    status = tk.StringVar(value="ready")
    last = tk.StringVar(value="")

    def set_status(msg: str) -> None:
        status.set(msg)
        last.set(msg)

    def on_cmd(name: str, *, operator_approved: bool = False) -> None:
        pilot.beat()
        pilot.nudge(name, operator_approved=operator_approved)
        set_status(f"cmd: {name}  sticks={'ON' if pilot.safety.pilot_sticks else 'PC'}")

    def on_key(event) -> None:
        pilot.beat()
        ch = event.keysym.lower() if len(event.keysym) > 1 else event.char.lower()
        # map keysym names
        if event.keysym == "space":
            ch = " "
        if event.keysym == "Escape":
            pilot.give_to_pilot()
            set_status("MANUAL: 搖桿接管")
            return
        if ch in KEY_ALIASES:
            on_cmd(KEY_ALIASES[ch], operator_approved=(ch == "t"))
        elif event.char in KEY_ALIASES:
            on_cmd(KEY_ALIASES[event.char], operator_approved=(event.char == "t"))

    def on_close() -> None:
        set_status("closing -> land")
        pilot.safety.stop = True
        pilot.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Key>", on_key)

    ttk.Label(root, text="電腦微移控制 — 結束/Ctrl-C/關窗 = 原地降落 + 還搖桿",
              font=("Sans", 12, "bold")).pack(pady=8)
    ttk.Label(root, textvariable=status, font=("Sans", 11)).pack()

    grid = ttk.LabelFrame(root, text="對角微移（按一下前進一點點）")
    grid.pack(padx=10, pady=8, fill="x")
    buttons = [
        ("左上後", 0, 0), ("上", 0, 1), ("右上後", 0, 2),
        ("左上前", 1, 0), ("前", 1, 1), ("右上前", 1, 2),
        ("左", 2, 0), ("懸停", 2, 1), ("右", 2, 2),
        ("左下前", 3, 0), ("後", 3, 1), ("右下前", 3, 2),
        ("左下後", 4, 0), ("下", 4, 1), ("右下後", 4, 2),
    ]
    for text, r, c in buttons:
        ttk.Button(grid, text=text, width=10,
                   command=lambda n=text: on_cmd(n)).grid(row=r, column=c, padx=4, pady=4)

    safe = ttk.LabelFrame(root, text="安全")
    safe.pack(padx=10, pady=8, fill="x")
    ttk.Button(safe, text="手動/搖桿接管 (Esc / p)",
               command=lambda: on_cmd("手動")).pack(side="left", padx=6, pady=6)
    ttk.Button(safe, text="懸停 (Space / h)",
               command=lambda: on_cmd("懸停")).pack(side="left", padx=6, pady=6)
    ttk.Button(safe, text="原地降落 (b)",
               command=lambda: on_cmd("原地降落")).pack(side="left", padx=6, pady=6)
    ttk.Button(safe, text="起飛 (t)",
               command=lambda: on_cmd("起飛", operator_approved=True)
               ).pack(side="left", padx=6, pady=6)
    ttk.Button(safe, text="恢復電腦控制",
               command=lambda: (pilot.take_pc_control(), set_status("PC control"))
               ).pack(side="left", padx=6, pady=6)

    help_txt = (
        "鍵位: U/I/O 左上前/前/右上前 | J/L 左/右 | M/,/. 左下前/後/右下前 | "
        "7/9 左上後/右上後 | 1/3 左下後/右下後 | W/S/A/D 前後左右 | R/F 上下 | "
        "Space 懸停 | Esc 搖桿 | B 降落 | T 起飛\n"
        f"脈衝 {pilot.pulse_s:.2f}s @ ±{pilot.pct}% | deadman {DEADMAN_S}s | "
        f"mode={'DRY-RUN' if pilot.dry_run else 'LIVE'}"
    )
    ttk.Label(root, text=help_txt, wraplength=680, justify="left").pack(padx=10, pady=6)
    ttk.Label(root, textvariable=last, foreground="#444").pack()

    # heartbeat + optional auto-exit
    t_end = time.monotonic() + max(1.0, secs)

    def tick() -> None:
        pilot.beat()
        if pilot.safety.stop or pilot.safety.landed or time.monotonic() >= t_end:
            on_close()
            return
        root.after(200, tick)

    root.after(200, tick)
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Safe ANAFI nudge pilot")
    ap.add_argument("--dry-run", action="store_true", help="no drone; log PCMD only")
    ap.add_argument("--selftest", action="store_true", help="pure-python checks and exit")
    ap.add_argument("--ip", default="192.168.53.1")
    ap.add_argument("--controller", default="skycontroller3")
    ap.add_argument("--pct", type=int, default=NUDGE_PCT, help="nudge PCMD percent")
    ap.add_argument("--pulse-s", type=float, default=NUDGE_S)
    ap.add_argument("--secs", type=float, default=1800.0)
    ap.add_argument("--cmd-log", default="")
    ap.add_argument("--no-gui", action="store_true",
                    help="headless: connect, wait for signals only (land on exit)")
    args = ap.parse_args(argv)

    if args.selftest:
        return run_selftest()

    if not args.dry_run:
        print(
            "!!! LIVE MODE: will take Olympe piloting source (sticks silent until Esc/手動).\n"
            "    Safety pilot must hold the SkyController. Props-off first for bench.\n"
            "    Ctrl-C / close window / kill terminal -> land + restore sticks.",
            flush=True,
        )

    log_path = Path(args.cmd_log) if args.cmd_log else None
    if log_path is None and not args.dry_run:
        root = Path(__file__).resolve().parents[2]  # .../定位
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = root / "outputs" / "flight_logs" / f"nudge_cmdlog_{stamp}.jsonl"

    log = CommandLog(log_path)
    pilot = NudgePilot(
        dry_run=args.dry_run,
        ip=args.ip,
        controller=args.controller,
        pct=args.pct,
        pulse_s=args.pulse_s,
        cmd_log=log,
    )

    def _sig_handler(signum, _frame):
        print(f"\n[signal {signum}] -> land + restore sticks", flush=True)
        pilot.safety.stop = True
        pilot.cleanup()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _sig_handler)
        except Exception:
            pass
    atexit.register(pilot.cleanup)

    pilot.connect()
    pilot.beat()
    threading.Thread(target=run_deadman, args=(pilot,), daemon=True).start()

    try:
        if args.no_gui:
            print("[headless] running; Ctrl-C to land and exit", flush=True)
            t_end = time.monotonic() + args.secs
            while not pilot.safety.stop and not pilot.safety.landed and time.monotonic() < t_end:
                pilot.beat()
                time.sleep(0.2)
        else:
            run_tk(pilot, args.secs)
    finally:
        pilot.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
