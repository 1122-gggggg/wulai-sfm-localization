#!/usr/bin/env python3
"""Ground bench for the stick-override safety mechanism. No drone, no flight.

Runs the project's REAL SkyControllerStickMonitor against the real SkyController
HID device and reports exactly what the backend would see:

  * that the joystick node is found and readable,
  * which axis crossed the deadzone,
  * that ``axes_active()`` -- the predicate that triggers
    ``_maybe_reclaim_from_sticks() -> give_to_pilot()`` -- fires.

The reclaim path has no airborne condition, so this exercises the whole
host-side chain except the final Olympe piloting-source call, on the ground.

    python tools/stick_override_bench.py [seconds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

OI = Path(__file__).resolve().parents[1] / "控制介面程式" / "operator_interface"
sys.path.insert(0, str(OI))

from olympe_live_backend import (  # noqa: E402
    SkyControllerStickMonitor,
    axes_active,
    find_skycontroller_joystick_path,
    _STICK_DEADZONE,
    _STICK_FLIGHT_AXES,
)

# Measured on a Parrot SkyController 3 v1.8.1 (2026-08-06 ground bench).
# Axes 0-3 are the two flight sticks; 4-5 are camera/gimbal controls and are
# deliberately NOT allowed to seize flight control from the PC.
AXIS_NAMES = {
    0: "左搖桿 X (yaw)",
    1: "左搖桿 Y (gaz)",
    2: "右搖桿 X (roll)",
    3: "右搖桿 Y (pitch)",
    4: "相機/雲台軸 (不奪取控制權)",
    5: "相機/肩鍵軸 (不奪取控制權)",
}


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    path = find_skycontroller_joystick_path()
    if path is None:
        print("[bench] FAIL: 找不到 SkyController joystick 節點 "
              "(/dev/input/by-id/usb-Parrot*Skycontroller*-joystick 或 /dev/input/js*)")
        print("[bench] 遙控器沒開或沒插 USB 時,backend 會進 STICK_MONITOR_FAIL,"
              "起飛會被 _takeoff_preflight 擋下。")
        return 1
    print(f"[bench] joystick 節點: {path}")
    print(f"[bench] deadzone = {_STICK_DEADZONE} (實測滿舵約 +/-28715, "
          f"即可用行程的 {_STICK_DEADZONE / 28715 * 100:.0f}%)")
    print(f"[bench] 可奪取控制權的飛行軸 = {list(_STICK_FLIGHT_AXES)}"
          "  (相機/雲台軸不列入)")

    fired: list[tuple[float, dict[int, int]]] = []
    t0 = time.monotonic()

    def on_active(axes: dict[int, int]) -> None:
        fired.append((time.monotonic() - t0, dict(axes)))
        moved = {a: v for a, v in axes.items() if abs(int(v)) > _STICK_DEADZONE}
        detail = ", ".join(
            f"{AXIS_NAMES.get(a, f'axis{a}')}={v:+6d}"
            f" ({abs(v) / 32767 * 100:.0f}%)"
            for a, v in sorted(moved.items())
        )
        print(f"[bench] {time.monotonic() - t0:5.1f}s  STICK_OVERRIDE -> "
              f"give_to_pilot()  |  {detail}", flush=True)

    def on_disconnect(reason: str) -> None:
        print(f"[bench] {time.monotonic() - t0:5.1f}s  DISCONNECT: {reason}  "
              "-> backend 會標記 STICK_MONITOR_FAIL,空中則原地降落", flush=True)

    mon = SkyControllerStickMonitor(on_active, on_disconnect=on_disconnect)
    if not mon.start():
        print("[bench] FAIL: stick monitor 啟動失敗")
        return 1
    print(f"[bench] device_name = {mon.device_name}")
    print(f"[bench] 請在 {seconds:.0f} 秒內撥動搖桿(超過 deadzone 才算)…\n")
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()

    print(f"\n[bench] 觸發次數: {len(fired)}")
    snapshot = mon.snapshot_axes()
    print(f"[bench] 最後軸值: {snapshot}")
    print(f"[bench] axes_active(最後軸值) = {axes_active(snapshot)}  "
          "(放開後應為 False)")
    if not fired:
        print("[bench] FAIL: 全程沒有觸發 —— 撥動幅度不足,或 HID 沒有事件")
        return 1
    print("[bench] PASS: 撥動搖桿確實會觸發交回控制權的判定")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
