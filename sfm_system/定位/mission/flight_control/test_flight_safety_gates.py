#!/usr/bin/env python3
"""Failure-injection tests for the real-flight safety gates (pure python, no drone).

Every test drives the REAL path_follow_flight.run_loop / SafetyMonitor /
command_to_body_percent / OlympePdrawGrabber logic with mock hooks. No olympe,
no torch, no GPU, no hardware.

Run:  pytest -q sfm_system/定位/mission/flight_control/test_flight_safety_gates.py
"""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import olympe_frame_source as ofs
import path_follow_flight as pff
import real_path_follow_controller as rpf

ZERO = (0, 0, 0, 0)


def run_ticks(max_ticks, pose_fn, *, route=((0.0, 0.0, 0.0), (8.0, 0.0, 0.0)),
              hooks_extra=None):
    """Run the real run_loop against mock hooks on a virtual clock.

    pose_fn(st) -> rpf.Pose | None; st has "t" (virtual now) and "tick".
    Returns (sent_pcmds, terminal_reason, log_records).
    """
    wp = [np.array(p, float) for p in route]
    ctrl = rpf.RouteAutoController(wp, poles=[],
                                   config=rpf.ControlConfig(inspect_waypoints=()))
    sent, records = [], []
    st = {"calls": 0, "t": 0.0, "tick": 0}

    def now():
        st["calls"] += 1
        if st["calls"] > max_ticks * 4:
            raise KeyboardInterrupt
        st["t"] += 1.0 / pff.CTRL_HZ
        return st["t"]

    def get_pose():
        st["tick"] += 1
        return pose_fn(st)

    extra = dict(hooks_extra or {})
    extra.setdefault("log_tick", records.append)
    hooks = pff.LoopHooks(
        get_pose=get_pose,
        olympe_yaw=lambda: None,
        send_pcmd=lambda r, p, y, g: sent.append((r, p, y, g)),
        now=now,
        **extra,
    )
    try:
        reason = pff.run_loop(hooks, ctrl, wp, verbose=False)
    except KeyboardInterrupt:
        reason = "tick cap"
    return sent, reason, records


def fresh_pose(st, x=0.0, y=0.0, z=0.0, yaw=0.0):
    return rpf.Pose(x=x, y=y, z=z, yaw=yaw, stamp=st["t"])


# ---------------------------------------------------------------------------
# Mode separation

def test_selftest_and_dry_run_never_import_olympe(tmp_path):
    pff.selftest()
    log = tmp_path / "dry_cmdlog.jsonl"
    state, progress = pff.dry_run(1, steps=300, cmd_log_path=str(log))
    assert "olympe" not in sys.modules, "--selftest/--dry-run must not touch Olympe"
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert lines and all(r.get("sink") == "dry-run" for r in lines)
    assert lines[-1].get("event") == "terminal"


def test_grab_only_source_never_arms():
    src = inspect.getsource(pff.grab_only)
    for token in ("TakeOff", "PCMD", "Landing", "Emergency", "moveBy", "moveTo"):
        assert token not in src, f"grab_only must never reference {token}"


def test_fly_is_the_only_arming_entrypoint():
    module_src = inspect.getsource(pff)
    fly_src = inspect.getsource(pff.fly)
    # every arming/movement Olympe message used by the module must live in fly()
    for token in ("TakeOff", "Emergency"):
        assert module_src.count(f"drone({token}") == fly_src.count(f"drone({token}")


# ---------------------------------------------------------------------------
# Frame / stream gates

def test_stream_lost_zero_then_land():
    def never_localize(st):
        raise AssertionError("stream gate must run before localization")
    sent, reason, records = run_ticks(
        400, never_localize, hooks_extra={"stream_healthy": lambda: False})
    assert sent and all(c == ZERO for c in sent)
    assert "stream lost" in reason
    assert records and all(r["blocked"] for r in records)
    assert any("stream" in r["reason"] for r in records)


def test_frozen_stream_goes_unhealthy():
    g = ofs.OlympePdrawGrabber.__new__(ofs.OlympePdrawGrabber)
    import threading
    g._lock = threading.Lock(); g._latest = None; g._stamp = 0.0; g._n = 0
    g._digest = None; g._dup_n = 0; g._frozen_warned = False
    g.stale_s = 5.0
    frame = (np.arange(720 * 1280 * 3, dtype=np.uint8) % 251).reshape(720, 1280, 3)
    for _ in range(ofs.FROZEN_DUP_FRAMES + 1):
        g._store(frame)
    assert g() is None, "frozen (duplicated) stream must yield no frame"
    assert not g.is_healthy(), "frozen stream must be unhealthy despite fresh stamps"
    g._store(frame.copy() + 1)
    assert g.is_healthy(), "a distinct new frame must recover the stream"


# ---------------------------------------------------------------------------
# Localization gates

def test_lost_pose_zero_then_land_never_emergency():
    sent, reason, _ = run_ticks(200, lambda st: None)
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"
    assert "EMERGENCY" not in reason


def test_weak_pose_hovers():
    sent, reason, _ = run_ticks(
        200, fresh_pose, hooks_extra={"pose_is_weak": lambda: True})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "low confidence -> land"


def test_low_inliers_hovers():
    sent, reason, _ = run_ticks(
        200, fresh_pose, hooks_extra={"pose_confidence": lambda: 10})
    assert sent and all(c == ZERO for c in sent)
    assert reason == "low confidence -> land"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pose_hovers(bad):
    sent, reason, _ = run_ticks(200, lambda st: fresh_pose(st, x=bad))
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"


def test_pose_jump_rejected_then_confirmed():
    def pose_fn(st):
        return fresh_pose(st) if st["tick"] <= 3 else fresh_pose(st, x=3.0)
    sent, _reason, records = run_ticks(20, pose_fn)
    assert sent[3] == ZERO, "first jumped fix must hover, not steer"
    assert any(r.get("jump_reject_u") for r in records), "jump must be logged"
    assert any(c != ZERO for c in sent[4:]), "confirmed relocation must resume driving"


def test_stale_pose_hovers():
    # stamps frozen in the past -> freshness gate blocks, hover, then land
    sent, reason, _ = run_ticks(200, lambda st: rpf.Pose(0.0, 0.0, 0.0, 0.0, stamp=-10.0))
    assert sent and all(c == ZERO for c in sent)
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# Route gates

def test_route_deviation_lands():
    sent, reason, _ = run_ticks(20, lambda st: fresh_pose(st, z=5.0))
    assert "route deviation" in reason
    assert sent[-1] == ZERO, "deviation abort must end on zero PCMD"


def test_route_completion_lands():
    sent, reason, _ = run_ticks(20, lambda st: fresh_pose(st, x=7.99))
    assert reason == "route complete -> land"
    assert sent[-1] == ZERO


def test_invalid_route_rejected(tmp_path):
    bad = tmp_path / "route.json"
    bad.write_text(json.dumps({"waypoints": [[0.0, 0.0, 0.0]]}))
    with pytest.raises(ValueError):
        rpf.load_waypoints(bad)


def test_heading_unavailable_is_none_before_motion():
    he = pff.HeadingEstimator()
    assert he.heading(None) is None, "no motion, no olympe yaw -> heading must be None"


# ---------------------------------------------------------------------------
# Safety switch / watchdog

def test_manual_sends_nothing():
    sent, _, records = run_ticks(30, fresh_pose,
                                 hooks_extra={"safety_poll": lambda: "MANUAL"})
    assert sent == [], "MANUAL: autonomy must not compete with the pilot's sticks"
    assert records and all(r["pcmd"] is None for r in records)


def test_hover_sends_zero():
    sent, _, _ = run_ticks(30, fresh_pose,
                           hooks_extra={"safety_poll": lambda: "HOVER"})
    assert sent and all(c == ZERO for c in sent)


def test_land_sends_zero_then_breaks():
    sent, reason, _ = run_ticks(30, fresh_pose,
                                hooks_extra={"safety_poll": lambda: "LAND"})
    assert sent == [ZERO], "LAND: exactly one zero PCMD before Landing"
    assert reason == "safety LAND command -> land"


def test_emergency_breaks_without_pcmd():
    sent, reason, _ = run_ticks(30, fresh_pose,
                                hooks_extra={"safety_poll": lambda: "EMERGENCY"})
    assert sent == [], "EMERGENCY must not stream PCMD; motor cut is the fly() finally's job"
    assert reason.startswith("EMERGENCY")


def test_watchdog_stall_after_beat_sends_zero():
    calls = []
    mon = pff.SafetyMonitor(lambda r, p, y, g: calls.append((r, p, y, g)),
                            None, timeout_s=0.06)
    mon.beat()                     # loop started, then stalls (no more beats)
    mon.start()
    time.sleep(0.3)
    mon.stop()
    assert ZERO in calls, "a stalled loop must be overridden with zero PCMD"


def test_no_nonzero_command_reuse_after_loss():
    # The loop may drive on last_good within the POSE_STALE_S freshness window,
    # but once it first hovers after the loss, no nonzero command may ever recur.
    def pose_fn(st):
        return fresh_pose(st) if st["tick"] <= 5 else None
    sent, reason, _ = run_ticks(200, pose_fn)
    assert any(c != ZERO for c in sent[:6]), "sanity: loop was actually driving first"
    first_zero_after_loss = next(i for i in range(6, len(sent)) if sent[i] == ZERO)
    assert all(c == ZERO for c in sent[first_zero_after_loss:]), \
        "after the loss is recognized, the previous nonzero command must never be reused"
    assert reason == "localization lost -> land"


# ---------------------------------------------------------------------------
# PCMD conversion

def test_pcmd_conversion_bounds():
    pose = rpf.Pose(0.0, 0.0, 0.0, yaw=0.3)
    for vx in (-100.0, -1.2, 0.0, 1.2, 100.0):
        for vy in (-50.0, 0.0, 50.0):
            for yt in (-3.0, 0.0, 3.0):
                cmd = rpf.Command("FOLLOW", np.array([vx, vy, 0.7]), yaw_target=yt,
                                  goal=np.zeros(3), path_error=0.0, progress=0.0)
                r, p, y, g = rpf.command_to_body_percent(cmd, pose)
                assert all(isinstance(v, int) for v in (r, p, y, g))
                assert r == 0 and 0 <= p <= 8 and abs(y) <= 25 and abs(g) <= 12
                assert all(-100 <= v <= 100 for v in (r, p, y, g))


def test_command_log_has_final_pcmd_and_reason():
    sent, _, records = run_ticks(10, fresh_pose)
    driven = [r for r in records if not r["blocked"]]
    assert driven, "driving ticks must be logged"
    for r in driven:
        assert r["pcmd"] is not None and len(r["pcmd"]) == 4
        assert "path_error_u" in r and "progress" in r and "action" in r
        assert tuple(r["pcmd"]) in sent, "logged PCMD must be exactly what was sent"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
