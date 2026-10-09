"""Read-only evidence checks for the operator's three-step preflight guide."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREFLIGHT_GUIDE_STEPS = ("compass", "map", "route")
PREFLIGHT_GUIDE_LABELS = {
    "compass": "羅盤校正狀態",
    "map": "地圖匯入",
    "route": "路線匯入／修改",
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

    def invalidate(self, step: str | None = None) -> str | None:
        """Drop one invalidated step and everything confirmed after it."""
        if step is None:
            if not self._evidence:
                return None
            step = next((s for s in PREFLIGHT_GUIDE_STEPS if s in self._evidence), None)
            if step is None:
                return None
        elif step not in self._evidence:
            return None
        index = PREFLIGHT_GUIDE_STEPS.index(step)
        for invalid in PREFLIGHT_GUIDE_STEPS[index:]:
            self._evidence.pop(invalid, None)
        return step

    def sync(self, current_evidence: dict[str, object | None]) -> str | None:
        """Drop one changed step and everything confirmed after it."""
        for step in PREFLIGHT_GUIDE_STEPS:
            if step not in self._evidence:
                break
            if current_evidence.get(step) == self._evidence[step]:
                continue
            return self.invalidate(step)
        return None


@dataclass(frozen=True)
class PreflightContext:
    """UI/backend facts needed by the read-only evidence checks."""

    live: bool
    map_points: object
    site_profile_path: Path | None
    route_snapshot: Any | None
    displayed_route_sha256: str | None
    route_point_count: int


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
    if getattr(state, "drone_magnetometer_started", None) is not False:
        return None, "等待飛機羅盤校正進行狀態讀回"
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
    note = "校正有效" if required == 0 else "沿用機上校正；韌體建議重新校正（非強制）"
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


def evaluate_preflight_step(
    step: str,
    state: object,
    context: PreflightContext,
    *,
    now: float,
    verify_route_hash: bool = False,
    memo: dict[tuple, tuple[object | None, str]] | None = None,
) -> tuple[object | None, str]:
    """Return immutable evidence for one human preflight confirmation."""
    memo_key = (step, id(state), id(context), verify_route_hash)
    if memo is not None and memo_key in memo:
        return memo[memo_key]
    if step == "compass":
        result = _compass_evidence(state, context.live)
    elif step == "map":
        result = _map_evidence(context)
    elif step == "route":
        result = _route_evidence(context, verify_route_hash)
    else:
        result = None, f"未知的起飛前步驟：{step}"
    if memo is not None:
        memo[memo_key] = result
    return result
