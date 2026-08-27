#!/usr/bin/env python3
"""Ground gravity / IMU attitude calibration helper.

Operator physically rotates the (props-off) airframe through three motions while
we sample Olympe-style attitude (roll, pitch, yaw in radians):

  1. YAW   — horizontal spin about gravity (keep deck roughly level)
  2. PITCH — tip nose forward / back
  3. ROLL  — tip left / right (sideways)

From the samples we report:
  - level tilt (mean roll/pitch during the yaw phase)
  - phase coverage (how much each Euler angle moved)
  - body-frame gravity unit vector consistency
  - pass/fail against simple thresholds

This does NOT arm motors. Live sampling is passive telemetry only.

Pure math + selftest (no drone):
  python gravity_calibration.py --selftest

Live guided session (SkyController / drone WiFi):
  python gravity_calibration.py --live --ip 192.168.53.1 --controller skycontroller3
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASES = ("yaw", "pitch", "roll")
PHASE_LABELS = {
    "yaw": "水平旋轉（繞重力軸轉一圈，機身保持水平）",
    "pitch": "前後俯仰（機頭抬高/壓低）",
    "roll": "左右側傾（往側邊翻）",
}

# Pass thresholds (tune after a few real benches)
MAX_LEVEL_TILT_DEG = 6.0  # allow the stable ~5° ANAFI attitude offset
MIN_YAW_SPAN_DEG = 90.0  # must cover enough yaw
MIN_PITCH_SPAN_DEG = 25.0
MIN_ROLL_SPAN_DEG = 25.0
MAX_G_RMS = 0.08  # unit-vector scatter of body gravity
MIN_SAMPLES_PER_PHASE = 15


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def attitude_to_body_gravity(roll: float, pitch: float) -> tuple[float, float, float]:
    """Body-frame unit gravity (aircraft NED convention used by ANAFI attitude).

    Gravity in NED is +Z (down). Body components:
      gx = -sin(pitch)
      gy =  sin(roll) * cos(pitch)
      gz =  cos(roll) * cos(pitch)
    so a level airframe yields ~ (0, 0, 1).
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    gx = -sp
    gy = sr * cp
    gz = cr * cp
    n = math.sqrt(gx * gx + gy * gy + gz * gz) or 1.0
    return gx / n, gy / n, gz / n


def _span_deg(values: list[float]) -> float:
    """Angular span in degrees, handling yaw wrap."""
    if not values:
        return 0.0
    # unwrap
    unwrapped = [values[0]]
    for v in values[1:]:
        prev = unwrapped[-1]
        d = (v - prev + math.pi) % (2 * math.pi) - math.pi
        unwrapped.append(prev + d)
    return math.degrees(max(unwrapped) - min(unwrapped))


@dataclass
class AttitudeSample:
    t_mono: float
    roll: float
    pitch: float
    yaw: float
    phase: str


@dataclass
class PhaseResult:
    phase: str
    n: int
    roll_mean_deg: float
    pitch_mean_deg: float
    yaw_span_deg: float
    roll_span_deg: float
    pitch_span_deg: float
    level_tilt_deg: float
    g_mean: tuple[float, float, float]
    g_rms: float
    ok: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class GravityCalResult:
    ok: bool
    phases: dict[str, PhaseResult]
    body_g_level: tuple[float, float, float] | None
    level_tilt_deg: float
    map_up_hint: str
    summary: str
    t_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(frozen=True)
class _PhaseMetrics:
    roll_mean: float
    pitch_mean: float
    yaw_span: float
    roll_span: float
    pitch_span: float
    level_tilt: float
    g_mean: tuple[float, float, float]
    g_rms: float


def _phase_metrics(samples: list[AttitudeSample]) -> _PhaseMetrics:
    rolls = [sample.roll for sample in samples]
    pitches = [sample.pitch for sample in samples]
    yaws = [sample.yaw for sample in samples]
    roll_mean = statistics.fmean(rolls)
    pitch_mean = statistics.fmean(pitches)
    yaw_span = _span_deg(yaws)
    roll_span = _span_deg(rolls)
    pitch_span = _span_deg(pitches)
    level_tilt = math.degrees(math.hypot(roll_mean, pitch_mean))

    gravity = [attitude_to_body_gravity(sample.roll, sample.pitch) for sample in samples]
    gx = statistics.fmean(value[0] for value in gravity)
    gy = statistics.fmean(value[1] for value in gravity)
    gz = statistics.fmean(value[2] for value in gravity)
    norm = math.sqrt(gx * gx + gy * gy + gz * gz) or 1.0
    g_mean = (gx / norm, gy / norm, gz / norm)
    dots = [
        max(
            -1.0,
            min(
                1.0,
                value[0] * g_mean[0] + value[1] * g_mean[1] + value[2] * g_mean[2],
            ),
        )
        for value in gravity
    ]
    angles = [math.acos(dot) for dot in dots]
    g_rms = math.sqrt(statistics.fmean(angle * angle for angle in angles))
    return _PhaseMetrics(
        roll_mean=roll_mean,
        pitch_mean=pitch_mean,
        yaw_span=yaw_span,
        roll_span=roll_span,
        pitch_span=pitch_span,
        level_tilt=level_tilt,
        g_mean=g_mean,
        g_rms=g_rms,
    )


def _phase_passes(
    phase: str,
    sample_count: int,
    metrics: _PhaseMetrics,
    notes: list[str],
) -> bool:
    ok = sample_count >= MIN_SAMPLES_PER_PHASE
    if phase == "yaw":
        if metrics.yaw_span < MIN_YAW_SPAN_DEG:
            ok = False
            notes.append(f"yaw span {metrics.yaw_span:.0f}° < {MIN_YAW_SPAN_DEG:.0f}°")
        if metrics.level_tilt > MAX_LEVEL_TILT_DEG:
            ok = False
            notes.append(f"level tilt {metrics.level_tilt:.1f}° > {MAX_LEVEL_TILT_DEG:.0f}°")
        if metrics.roll_span > 40 or metrics.pitch_span > 40:
            notes.append(
                f"yaw phase not level-ish "
                f"(roll_span={metrics.roll_span:.0f}° "
                f"pitch_span={metrics.pitch_span:.0f}°)"
            )
        if metrics.g_rms > MAX_G_RMS:
            ok = False
            notes.append(f"gravity scatter rms {metrics.g_rms:.3f} rad > {MAX_G_RMS}")
    elif phase == "pitch" and metrics.pitch_span < MIN_PITCH_SPAN_DEG:
        ok = False
        notes.append(f"pitch span {metrics.pitch_span:.0f}° < {MIN_PITCH_SPAN_DEG:.0f}°")
    elif phase == "roll" and metrics.roll_span < MIN_ROLL_SPAN_DEG:
        ok = False
        notes.append(f"roll span {metrics.roll_span:.0f}° < {MIN_ROLL_SPAN_DEG:.0f}°")
    return ok


class GravityCalibrator:
    """Collect attitude samples in three guided phases and score them."""

    def __init__(self) -> None:
        self.phase: str | None = None
        self.samples: list[AttitudeSample] = []
        self.started = False
        self.finished = False
        self._phase_order: list[str] = []

    def reset(self) -> None:
        self.__init__()

    def start(self) -> None:
        self.reset()
        self.started = True
        self.begin_phase("yaw")

    def begin_phase(self, phase: str) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase}")
        self.phase = phase
        if phase not in self._phase_order:
            self._phase_order.append(phase)

    def next_phase(self) -> str | None:
        """Advance yaw -> pitch -> roll -> done. Returns new phase or None if finished."""
        if self.phase is None:
            self.begin_phase("yaw")
            return "yaw"
        try:
            i = PHASES.index(self.phase)
        except ValueError:
            return None
        if i + 1 >= len(PHASES):
            self.phase = None
            self.finished = True
            return None
        nxt = PHASES[i + 1]
        self.begin_phase(nxt)
        return nxt

    def add_sample(
        self, roll: float, pitch: float, yaw: float, t_mono: float | None = None
    ) -> None:
        if not self.started or self.phase is None or self.finished:
            return
        if not all(_finite(v) for v in (roll, pitch, yaw)):
            return
        self.samples.append(
            AttitudeSample(
                t_mono=time.monotonic() if t_mono is None else float(t_mono),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
                phase=self.phase,
            )
        )

    def samples_for(self, phase: str) -> list[AttitudeSample]:
        return [s for s in self.samples if s.phase == phase]

    def analyze_phase(self, phase: str) -> PhaseResult:
        ss = self.samples_for(phase)
        notes: list[str] = []
        if len(ss) < MIN_SAMPLES_PER_PHASE:
            notes.append(f"samples {len(ss)} < {MIN_SAMPLES_PER_PHASE}")
        if not ss:
            return PhaseResult(
                phase=phase,
                n=0,
                roll_mean_deg=float("nan"),
                pitch_mean_deg=float("nan"),
                yaw_span_deg=0.0,
                roll_span_deg=0.0,
                pitch_span_deg=0.0,
                level_tilt_deg=float("nan"),
                g_mean=(0.0, 0.0, 1.0),
                g_rms=float("nan"),
                ok=False,
                notes=notes + ["no samples"],
            )

        metrics = _phase_metrics(ss)
        ok = _phase_passes(phase, len(ss), metrics, notes)

        return PhaseResult(
            phase=phase,
            n=len(ss),
            roll_mean_deg=math.degrees(metrics.roll_mean),
            pitch_mean_deg=math.degrees(metrics.pitch_mean),
            yaw_span_deg=metrics.yaw_span,
            roll_span_deg=metrics.roll_span,
            pitch_span_deg=metrics.pitch_span,
            level_tilt_deg=metrics.level_tilt,
            g_mean=metrics.g_mean,
            g_rms=metrics.g_rms,
            ok=ok,
            notes=notes,
        )

    def result(self) -> GravityCalResult:
        phases = {p: self.analyze_phase(p) for p in PHASES}
        yaw_r = phases["yaw"]
        body_g = yaw_r.g_mean if yaw_r.n else None
        level_tilt = yaw_r.level_tilt_deg if yaw_r.n else float("nan")
        ok = all(phases[p].ok for p in PHASES if phases[p].n > 0) and all(
            phases[p].n >= MIN_SAMPLES_PER_PHASE for p in PHASES
        )
        # Map-frame hint for this project: GLOMAP gravity-up = -Y
        map_up = (
            "GLOMAP map-up is -Y. Body gravity (NED down) from level phase is "
            f"g_body≈({body_g[0]:+.3f},{body_g[1]:+.3f},{body_g[2]:+.3f}); "
            "after hover+PnP, camera -Y_map should align with world up "
            "(opposite IMU down within a few degrees)."
            if body_g
            else "insufficient level samples"
        )
        parts = []
        for p in PHASES:
            r = phases[p]
            parts.append(
                f"{p}: n={r.n} yawΔ={r.yaw_span_deg:.0f}° pitchΔ={r.pitch_span_deg:.0f}° "
                f"rollΔ={r.roll_span_deg:.0f}° tilt={r.level_tilt_deg:.1f}° "
                f"g_rms={r.g_rms:.3f} {'OK' if r.ok else 'FAIL'}"
                + (f" ({'; '.join(r.notes)})" if r.notes else "")
            )
        summary = f"{'PASS' if ok else 'FAIL'} | level_tilt={level_tilt:.1f}° | " + " || ".join(
            parts
        )
        return GravityCalResult(
            ok=ok,
            phases=phases,
            body_g_level=body_g,
            level_tilt_deg=level_tilt,
            map_up_hint=map_up,
            summary=summary,
        )

    def to_jsonable(self) -> dict[str, Any]:
        res = self.result()
        out = {
            "ok": res.ok,
            "summary": res.summary,
            "level_tilt_deg": res.level_tilt_deg,
            "body_g_level": res.body_g_level,
            "map_up_hint": res.map_up_hint,
            "t_iso": res.t_iso,
            "n_samples": len(self.samples),
            "phases": {},
        }
        for p, pr in res.phases.items():
            d = asdict(pr)
            out["phases"][p] = d
        return out


# ---------------------------------------------------------------------------
# Self-test with synthetic motions


def _selftest() -> int:
    cal = GravityCalibrator()
    cal.start()
    # yaw phase: level, spin 360°
    for i in range(40):
        yaw = i / 40 * 2 * math.pi
        cal.add_sample(roll=0.01, pitch=-0.02, yaw=yaw, t_mono=i * 0.05)
    assert cal.next_phase() == "pitch"
    for i in range(30):
        pitch = math.radians(-30 + i * 2)  # -30 .. +28
        cal.add_sample(roll=0.0, pitch=pitch, yaw=0.5, t_mono=10 + i * 0.05)
    assert cal.next_phase() == "roll"
    for i in range(30):
        roll = math.radians(-35 + i * 2.5)
        cal.add_sample(roll=roll, pitch=0.0, yaw=0.5, t_mono=20 + i * 0.05)
    assert cal.next_phase() is None
    res = cal.result()
    assert res.ok, res.summary
    assert res.level_tilt_deg < 3.0, res.level_tilt_deg
    assert res.phases["yaw"].yaw_span_deg >= 300
    assert res.phases["pitch"].pitch_span_deg >= 40
    assert res.phases["roll"].roll_span_deg >= 40

    # failing case: no motion
    bad = GravityCalibrator()
    bad.start()
    for i in range(20):
        bad.add_sample(0.0, 0.0, 0.1, t_mono=i * 0.05)
    assert not bad.analyze_phase("yaw").ok

    # body gravity level
    g = attitude_to_body_gravity(0.0, 0.0)
    assert abs(g[0]) < 1e-9 and abs(g[1]) < 1e-9 and abs(g[2] - 1.0) < 1e-9

    print("gravity_calibration selftest OK")
    return 0


def _run_live(ip: str, controller: str, out: Path | None) -> int:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import olympe_frame_source as ofs

    print("Connecting (passive). PROPS OFF. Follow on-screen phases.", flush=True)
    drone = ofs.connect(ip, controller=controller)
    cal = GravityCalibrator()
    cal.start()
    print(f"\n=== PHASE yaw: {PHASE_LABELS['yaw']} ===", flush=True)
    print("Rotate the aircraft horizontally. Press Enter when done.", flush=True)

    stop = {"f": False}

    def reader():
        from olympe.messages.ardrone3.PilotingState import AttitudeChanged

        while not stop["f"]:
            try:
                st = drone.get_state(AttitudeChanged)
            except Exception:
                st = None
            if st:
                cal.add_sample(
                    float(st.get("roll", 0.0)),
                    float(st.get("pitch", 0.0)),
                    float(st.get("yaw", 0.0)),
                )
            time.sleep(0.05)

    import threading

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        for phase in PHASES:
            if cal.phase != phase:
                cal.begin_phase(phase)
            print(f"\n>>> {PHASE_LABELS[phase]}", flush=True)
            input("    (rotate now; Enter to next phase) ")
            n = len(cal.samples_for(phase))
            print(f"    collected {n} samples", flush=True)
            if phase != "roll":
                cal.next_phase()
            else:
                cal.finished = True
                cal.phase = None
    finally:
        stop["f"] = True
        th.join(timeout=1.0)
        try:
            drone.disconnect()
        except Exception:
            pass

    res = cal.to_jsonable()
    print("\n=== RESULT ===")
    print(res["summary"])
    print(res["map_up_hint"])
    if out is None:
        root = Path(__file__).resolve().parents[2]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = root / "outputs" / "flight_logs" / f"gravity_cal_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0 if res["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ANAFI gravity / attitude calibration")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--ip", default="192.168.53.1")
    ap.add_argument("--controller", default="skycontroller3")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.live:
        return _run_live(args.ip, args.controller, Path(args.out) if args.out else None)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
