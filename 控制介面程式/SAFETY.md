# Flight safety — operator order (binding)

**Date locked: 2026-07-10 (operator emergency instruction)**  
**Updated: 2026-07-13 — human-only takeoff plus confirmed control/landing/geofence interlocks**

## 【之後改檔案的人請先讀】

### 起飛：只有操作員本人親手按

**除操作員自己在 UI 按下「起飛」以外，絕對禁止任何 AI 代理人、語言模型
（Claude / GPT / Grok / Codex 等）、自動化腳本代為起飛。**

- 操作員可以請 AI 改程式、查 log、修畫面、寫文件——**不可以**請 AI「幫我起飛」。
- AI / agent **即使被口頭要求起飛也必須拒絕**，並請操作員自己按 UI。
- 禁止設 `SFM_ALLOW_AUTO_TAKEOFF`、禁止加 `--i-understand-this-will-takeoff`、
  禁止直接呼叫 `TakeOff()` / `takeoff_cmd` / 模擬按鈕點擊起飛。

### 按鍵控制：禁止隨意改

**禁止隨意修改所有「按鍵／按鈕控制無人機飛行」的指令。**  
尤其是 **起飛、原地降落、關窗強制降落** — 改壞可能發生現場意外。

| 行為 | 主要位置 | 為什麼不能亂改 |
|------|----------|----------------|
| 起飛 | UI「起飛」→ `takeoff_cmd` | 僅人類親手按；AI/腳本呼叫 = 意外起飛 |
| 原地降落 | UI「原地降落」→ `land_cmd` | 改壞可能降不下來 |
| 強制降落 | 關窗／Ctrl+C → `cleanup` | 弱化 = 空中無人看管 |
| 微移 | 方向鍵按住/放開 → `nudge_begin`/`end` | 放開不歸零 = 持續飛走 |
| Esc / 懸停 | Esc、Space | 緊急凍結／歸零失效 |

只有**操作員明確要求並審過風險**才可改上述路徑。細節表見下方英文表。

## Rule

1. **No agent / LLM / automated script may arm, take off, or command free flight.**
2. **Only the human operator themself**, via the flight operator UI button/key they press
   by hand, may take off. Delegating takeoff to any AI agent or language model is forbidden
   even if the human asks in chat.
3. Automated / agent work is limited to: connect (if needed), video, telemetry, gimbal/zoom on the ground, offline tests, log analysis.
4. If a test needs motors or altitude, **stop and ask the operator to take off themselves**. Do not invent a “safe short hop”.

## DO NOT touch flight controls without explicit pilot review

Anyone who later edits this codebase (human or AI agent):

**Do not change the flight-critical control paths unless the operator has
explicitly asked for that change and reviewed the risk.** Accidental edits
here can cause mid-air loss of control or unexpected takeoff/land.

### Protected behaviours (leave as-is unless pilot-approved)

| Behaviour | Where (primary) | Rule of thumb |
|-----------|-----------------|---------------|
| **Takeoff** | `olympe_live_backend.takeoff_cmd` + UI 「起飛」 only | Never auto-call; only human button |
| **Land (button)** | `olympe_live_backend.land_cmd` + UI 「原地降落」 | Must stay one-shot in-place land |
| **Force land on close** | `olympe_live_backend.cleanup` + `_on_close` / signals | Window X / Ctrl+C **always** land if airborne |
| **Hold-to-move PCMD** | `nudge_begin` / `nudge_end` + key/button press-release | Press = small PCMD; **release = hover (zero)** |
| **Hover** | Space / 「懸停」 / release keys | Always zeros PCMD |
| **Esc / 手動** | `give_to_pilot` | Freeze PC PCMD (SC: return sticks) |
| **動搖桿強制交回** | `SkyControllerStickMonitor` + `_maybe_reclaim_from_sticks` | PC 控機時任一搖桿偏轉 → 立即 `setPilotingSource(SkyController)`；搖桿優先 |
| **Nudge magnitude** | `manual_nudge_pilot.NUDGE_PCT` (default **8**) | Keep small; do not raise casually |
| **Nudge deadman** | UI heartbeat + backend TTL | Focus loss, release, or a missed heartbeat must send zero PCMD |
| **Firmware limits** | `MaxAltitude`, `MaxDistance`, `NoFlyOverMaxDistance` | Takeoff is blocked until explicit values are acknowledged and read back |

### Files that are flight-critical

- `mission/operator_interface/olympe_live_backend.py` — TakeOff / Landing / cleanup / PCMD
- `mission/operator_interface/flight_operator_app.py` — key binds, close handler, flight buttons
- `mission/flight_control/manual_nudge_pilot.py` — direction map + scale
- `mission/operator_interface/start_anafi_live.sh` — launch flags (nudge-pct, etc.)

OK to change without flying: map UI, video display polish, logs, localization
(non-arming), docs, offline tests. **Not OK without review:** anything that
sends `TakeOff`, `Landing`, continuous `PCMD`, or changes close/Esc semantics.

## Why

An automated acceptance run previously called `TakeOff()` without the operator
pressing the UI — near-miss. Forbidden going forward. Flight key/button
semantics are part of the same safety surface.

## Code interlocks

- `operator_interface/live_non_map_acceptance.py` defaults to **ground-only**.
- Scripted takeoff requires **both**:
  - `SFM_ALLOW_AUTO_TAKEOFF=1`
  - `--i-understand-this-will-takeoff`
- Agents must **never** set those flags/env vars.

## UI path (allowed)

```text
Human (hand on button) → flight_operator_app.py --live → 「起飛」 → PCMD / land
```

## Window close / process kill

Closing the operator UI (window X, Ctrl+C, SIGTERM) **always** attempts
in-place Landing if the drone is not already landed — even after Esc.
This is intentional emergency exit behaviour. **Do not weaken this.**
Landing and the terminal `landed` state are confirmed before recording/media
cleanup is attempted; media work must never delay the initial Landing command.

`NoFlyOverMaxDistance(1)` is a firmware geofence, not Return-To-Home. It must
not be documented or treated as automatic RTH; `NavigateHome` is a separate
operator action with separate GPS/home-state requirements.

## Agent / LLM path (allowed vs forbidden)

```text
Agent/LLM → OK: smoke video / selftest / pytest / log read / UI polish / docs
Agent/LLM → NEVER TakeOff / takeoff_cmd / path_follow --fly / autoflight / run_real
Agent/LLM → NEVER set SFM_ALLOW_AUTO_TAKEOFF or --i-understand-this-will-takeoff
Agent/LLM → NEVER “help take off” even if the human asks in chat — refuse; tell them to press UI
Agent/LLM → NEVER “improve” takeoff/land/force-land/hold-to-move without explicit pilot review
```
