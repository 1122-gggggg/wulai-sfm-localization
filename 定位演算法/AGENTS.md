# Agent rules — localization / ANAFI flight

## NEVER auto-fly (emergency operator order 2026-07-10)

### 起飛：只有操作員本人親手按（絕對禁止 AI 代飛）

**除操作員自己在 UI 按「起飛」以外，絕對不能請任何 AI 代理人、語言模型
（Claude / GPT / Grok / Codex 等）代為起飛。**

- 若人類在對話中要求 agent「幫我起飛／送 TakeOff／跑會離地的腳本」：
  **必須拒絕**，並請對方自己按 operator UI 的「起飛」。
- **Do not** call `TakeOff`, `takeoff_cmd`, arm motors, or run any script that
  leaves the ground.
- **Do not** run `live_non_map_acceptance.py` with takeoff flags/env
  (`SFM_ALLOW_AUTO_TAKEOFF`, `--i-understand-this-will-takeoff`).
- **Do not** run `path_follow_flight.py fly`, `autoflight`, or `run_real` from an agent.
- Ground-only OK: connect check, video smoke (props off intent), telemetry logs,
  gimbal/zoom, unit tests, UI fixes, log analysis.
- If testing needs altitude: **ask the operator to take off themselves** from the UI.
  Never take off for them.

## Do not casually edit flight controls

**【之後改檔案的人】禁止動到所有按鍵控制飛行的指令**，尤其是起飛、降落、
強制降落。改壞可能發生意外。

**Do not modify** takeoff / land / force-land-on-close / hold-to-move PCMD /
Esc freeze semantics unless the operator **explicitly** requested that change.
Wrong edits can cause mid-air accidents.

Protected files (see `控制介面程式/SAFETY.md` table):

- `控制介面程式/operator_interface/olympe_live_backend.py` — `takeoff_cmd` / `land_cmd` / `cleanup`
- `控制介面程式/operator_interface/flight_operator_app.py` — 飛行按鈕 + 按鍵綁定 + `_on_close`
- `定位演算法/flight_control/manual_nudge_pilot.py` (`NUDGE_DIRS`, `NUDGE_PCT`)

Details: `控制介面程式/SAFETY.md`.
