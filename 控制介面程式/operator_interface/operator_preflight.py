"""Read-only evidence checks for the operator's four-step preflight guide."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREFLIGHT_GUIDE_STEPS = ("compass", "map", "route", "system")
PREFLIGHT_GUIDE_LABELS = {
    "compass": "羅盤校正狀態",
    "map": "地圖匯入",
    "route": "路線匯入／修改",
    "system": "串流與系統資訊",
}


class SequentialPreflightGuide:
    """Human confirmations that must be completed in one fixed order."""

    def __init__(self) -> None:
        self._evidence: dict[str, object] = {}

    @property
    def current_step(self) -> str | None:
        return next(
            (step for step in PREFLIGHT_GUIDE_STEPS if step not in self._evidence),
            None,
        )

    @property
    def complete(self) -> bool:
        return self.current_step is None

    @property
    def confirmed_steps(self) -> tuple[str, ...]:
        return tuple(
            step for step in PREFLIGHT_GUIDE_STEPS if step in self._evidence
        )

    def confirm_current(self, evidence: object) -> str:
        step = self.current_step
        if step is None:
            raise ValueError("preflight guide is already complete")
        if evidence is None:
            raise ValueError(f"preflight step {step} has no valid evidence")
        self._evidence[step] = evidence
        return step

    def sync(self, current_evidence: dict[str, object | None]) -> str | None:
        """Drop one changed step and everything confirmed after it."""
        for index, step in enumerate(PREFLIGHT_GUIDE_STEPS):
            if step not in self._evidence:
                break
            if current_evidence.get(step) == self._evidence[step]:
                continue
            for invalid in PREFLIGHT_GUIDE_STEPS[index:]:
                self._evidence.pop(invalid, None)
            return step
        return None


@dataclass(frozen=True)
class PreflightContext:
    """UI/backend facts needed by the read-only evidence checks."""

    live: bool
    map_points: object
    site_profile_path: Path | None
    route_snapshot: Any | None
    displayed_route_sha256: str | None
    route_visible: bool
    route_point_count: int
    video_frame_available: bool
    stream_last_stamp: object | None
    video_frame_stamp: object | None
    min_takeoff_battery_pct: object
    via_controller: bool
    safety_log_durable: bool | None
    safety_log_healthy: bool | None
    inventory_takeoff_ready: bool | None
    inventory_block_reason: str


def _compass_evidence(state: object, live: bool) -> tuple[object | None, str]:
    if not live:
        return ("compass", "simulated"), "SIM 模式不提供韌體羅盤校正"
    flight_state = str(getattr(state, "flight_state", "") or "")
    if flight_state.rsplit(".", 1)[-1].lower() != "landed":
        return None, "必須先確認飛機為 landed"
    required = getattr(state, "drone_magnetometer_required", None)
    if required is None:
        return None, "等待飛機羅盤狀態讀回"
    if getattr(state, "drone_magnetometer_failed", None) is True:
        return None, "飛機羅盤校正失敗，請重新校正"
    if getattr(state, "drone_magnetometer_started", None) is True:
        return None, "飛機羅盤校正仍在進行"
    if required == 1:
        return None, "飛機要求完成羅盤校正"
    if required not in {0, 2}:
        return None, f"未知的飛機羅盤狀態：{required!r}"
    controller_raw = str(
        getattr(state, "skycontroller_magnetometer_state", "unknown")
        or "unknown"
    )
    controller_key = controller_raw.rsplit(".", 1)[-1].replace("_", "").lower()
    if controller_key.startswith("calibrating"):
        return None, "SkyController 羅盤校正仍在進行"
    note = "校正有效" if required == 0 else "韌體建議校正，請人工確認現場條件"
    return ("compass", int(required)), note


def _map_evidence(context: PreflightContext) -> tuple[object | None, str]:
    try:
        point_count = len(context.map_points)  # type: ignore[arg-type]
    except TypeError:
        point_count = 0
    if point_count <= 0:
        return None, "尚未載入可用地圖點雲"
    profile_path = context.site_profile_path
    if context.live and profile_path is None:
        return None, "真機模式尚未綁定場域設定"
    if profile_path is not None and not profile_path.is_file():
        return None, "場域設定檔已不存在"
    identity = str(profile_path.resolve()) if profile_path is not None else "simulated-map"
    return (
        "map",
        identity,
        id(context.map_points),
        point_count,
    ), f"已載入 {point_count} 個地圖點，請確認場域正確"


def _route_evidence(
    context: PreflightContext,
    verify_route_hash: bool,
) -> tuple[object | None, str]:
    snapshot = context.route_snapshot
    if snapshot is None:
        return None, "尚未選定通過驗證的飛行路線"
    if context.displayed_route_sha256 != snapshot.sha256:
        return None, "畫面路線與任務路線不一致"
    if not context.route_visible:
        return None, "請先顯示規劃路徑並逐點檢查"
    if context.route_point_count != len(snapshot.waypoints):
        return None, "畫面航點數與任務路線不一致"
    try:
        stat = snapshot.path.stat()
        if verify_route_hash:
            snapshot.verify_file_unchanged()
    except (OSError, ValueError) as exc:
        return None, f"路線檔驗證失敗：{exc}"
    return (
        "route",
        str(snapshot.path),
        snapshot.sha256,
        snapshot.site_id,
        snapshot.coordinate_frame_id,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    ), f"{len(snapshot.waypoints)} 個航點，請確認順序、高度與轉向位置"


def _fresh_stream_stamp(context: PreflightContext) -> object | None:
    if context.stream_last_stamp not in (None, 0):
        return context.stream_last_stamp
    return context.video_frame_stamp


def _live_state_problem(state: object) -> str | None:
    flight_state = str(getattr(state, "flight_state", "") or "")
    if flight_state.rsplit(".", 1)[-1].lower() != "landed":
        return "飛機狀態必須為 landed"
    if not bool(getattr(state, "link_ok", False)):
        return "Olympe 控制連線異常"
    incident = str(getattr(state, "active_incident", "") or "")
    if incident:
        return f"仍有安全事件：{incident}"
    alert = str(getattr(state, "alert_state", "") or "")
    alert_key = alert.rsplit(".", 1)[-1].replace("_", "").lower()
    if alert_key not in {"none", "noalert"}:
        return f"飛控警示尚未清除：{alert or 'unknown'}"
    return None


def _telemetry_problem(state: object, now: float) -> str | None:
    telemetry_ns = getattr(state, "telemetry_read_mono_ns", None)
    if telemetry_ns is None:
        return "等待即時遙測讀回"
    telemetry_age_s = now - int(telemetry_ns) / 1_000_000_000.0
    if telemetry_age_s < 0.0 or telemetry_age_s > 2.0:
        return f"遙測已過期（{max(0.0, telemetry_age_s):.1f}s）"
    return None


def _stream_problem(
    state: object,
    context: PreflightContext,
    now: float,
) -> str | None:
    try:
        stream_age_s = now - float(_fresh_stream_stamp(context))
    except (TypeError, ValueError):
        return "等待即時串流影格"
    if (
        not context.video_frame_available
        or stream_age_s < 0.0
        or stream_age_s > 1.0
    ):
        return f"串流影格不新鮮（{max(0.0, stream_age_s):.1f}s）"
    stream = str(getattr(state, "stream", ""))
    if stream.upper() not in {"PREVIEW", "OK"}:
        return f"串流狀態尚未就緒：{stream or '?'}"
    return None


def _system_readiness_problem(
    state: object,
    context: PreflightContext,
) -> str | None:
    try:
        battery = float(getattr(state, "battery_pct", -1.0))
        battery_floor = float(context.min_takeoff_battery_pct)
    except (TypeError, ValueError):
        return "電量讀回或起飛門檻格式無效"
    if not math.isfinite(battery) or battery < battery_floor:
        return f"電量低於起飛門檻 {battery_floor:.0f}%"
    if context.via_controller and not bool(getattr(state, "stick_monitor_ok", False)):
        return "SkyController 搖桿監看尚未就緒"
    if context.safety_log_durable is False or context.safety_log_healthy is False:
        return "安全紀錄尚未確認可寫入"
    if context.inventory_takeoff_ready is False:
        return context.inventory_block_reason
    return None


def _system_evidence(
    state: object,
    context: PreflightContext,
    now: float,
) -> tuple[object | None, str]:
    if not context.live:
        if not context.video_frame_available:
            return None, "等待模擬串流畫面"
        return ("system", "simulated"), "模擬串流已顯示"
    problem = _live_state_problem(state)
    if problem is not None:
        return None, problem
    problem = _telemetry_problem(state, now)
    if problem is not None:
        return None, problem
    problem = _stream_problem(state, context, now)
    if problem is not None:
        return None, problem
    problem = _system_readiness_problem(state, context)
    if problem is not None:
        return None, problem
    return (
        "system",
        "live",
    ), "串流、遙測、連線與電量正常；GPS／高度／距離限制僅提示，不阻擋起飛"


def evaluate_preflight_step(
    step: str,
    state: object,
    context: PreflightContext,
    *,
    now: float,
    verify_route_hash: bool = False,
) -> tuple[object | None, str]:
    """Return immutable evidence for one human preflight confirmation."""
    if step == "compass":
        return _compass_evidence(state, context.live)
    if step == "map":
        return _map_evidence(context)
    if step == "route":
        return _route_evidence(context, verify_route_hash)
    if step == "system":
        return _system_evidence(state, context, now)
    return None, f"未知的起飛前步驟：{step}"
