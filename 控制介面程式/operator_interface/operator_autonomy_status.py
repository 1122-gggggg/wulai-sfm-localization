"""Operator text for route phases; no flight commands."""

import math


def _blocked_leg_stage(label: str, lowered: str) -> tuple[str, str]:
    if "waiting for horizontal stop" in lowered:
        return "turn_brake", f"水平減速後修正朝向（{label}）"
    if "speed" in lowered:
        return "speed_guard", f"限速懸停中（{label}）"
    if "paused" in lowered:
        return "paused", f"已暫停（{label}）"
    if (
        "no fresh pose" in lowered
        or "low confidence" in lowered
        or "localization recovery" in lowered
    ):
        return "wait_pose", f"等待可靠定位（{label}）"
    if "map constraint" in lowered:
        return "wait_pose", f"地圖定位中斷，重新定位中（{label}）"
    if "localization recovered" in lowered:
        return "pose_recovered", f"定位已恢復，準備繼續{label}"
    return "blocked", f"受阻懸停中（{label}）"


def _phase_leg_stage(phase: object, label: str) -> tuple[str, str] | None:
    stages = {
        "turn": ("turn", f"旋轉機頭朝向{label}"),
        "turn_drift_recovery": ("turn_drift_recovery", f"修正風漂移後轉向（{label}）"),
        "turn_settle": ("turn_settle", f"確認水平穩定後轉向（{label}）"),
        "turn_recovery_wait": ("turn_recovery_wait", f"等待可靠定位與速度以修正漂移（{label}）"),
        "turn_brake": ("turn_brake", f"水平減速後轉向（{label}）"),
        "yaw_alignment_hold": ("align_hold", f"機頭對準保持確認中（{label}）"),
        "yaw_alignment_confirmed": ("aligned", f"已朝向{label}"),
        "translate": ("enroute", f"前往{label}"),
        "height_adjust": ("height_adjust", f"修正高度中（{label}）"),
        "route_rejoin": ("route_rejoin", f"回到航段中（{label}）"),
        "waypoint_centering": ("waypoint_centering", f"位置與高度調整中（{label}）"),
        "yaw_alignment_timeout": ("align_timeout", f"對頭逾時保持（{label}）"),
        "final_centering": ("final_centering", f"終點置中（{label}）"),
    }
    return stages.get(phase) if isinstance(phase, str) else None


def _reason_leg_stage(label: str, lowered: str, phase: object, last: dict) -> tuple[str, str]:
    if "pose jump" in lowered:
        return "jump_pause", f"定位跳變暫停（{label}）・需人工繼續"
    if "stick override" in lowered:
        return "stick", f"搖桿接管懸停（{label}）"
    if "stream" in lowered and ("stale" in lowered or "lost" in lowered):
        return "stream", f"串流中斷懸停中（{label}）"
    if (
        "no fresh pose" in lowered
        or "localization recovery" in lowered
        or "heading unavailable" in lowered
    ):
        return "wait_pose", f"等定位懸停中（{label}）"
    if "speed" in lowered:
        return "speed_guard", f"限速懸停中（{label}）"
    if "safety" in lowered:
        return "safety", f"安全懸停中（{label}）"
    if last.get("blocked"):
        return "blocked", f"受阻懸停中（{label}）"
    if phase == "idle":
        return "idle_hold", f"待命中（{label}）"
    return "hover", f"懸停中（{label}）"


def leg_stage(label: str, last: dict) -> tuple[str, str]:
    """Derive one stable stage key plus operator text for a route tick."""
    phase = last.get("phase")
    action = str(last.get("action") or "")
    reason = str(last.get("reason") or "")
    lowered = reason.lower()
    arrival = _arrival_stage(label, action)
    if arrival is not None:
        return arrival
    if last.get("blocked"):
        return _blocked_leg_stage(label, lowered)
    if not last.get("has_pose", True):
        return "searching_pose", f"找尋{label}中"
    if "final hold" in action:
        return "final_hold", f"終點{label}到達確認中"
    if "pending inspection" in action or action.startswith("ABORT"):
        return "abort", f"巡檢缺失中止降落（{label}）"
    if action.startswith("LAND") or "final path reached" in action:
        return "landing", f"終點到達・降落中（{label}）"
    if "INSPECT" in action:
        return "inspect", f"巡檢對準／擷取中（{label}）"
    if "yaw localization search" in reason:
        suffix = _search_suffix(last)
        return "yaw_search", f"找定位旋轉中（{label}）{suffix}"
    phase_stage = _phase_leg_stage(phase, label)
    if phase_stage is not None:
        return phase_stage
    return _reason_leg_stage(label, lowered, phase, last)


def _search_suffix(last: dict) -> str:
    deg = last.get("search_deg")
    try:
        suffix = (
            f"（已旋轉{float(deg):.0f}°）" if deg is not None and math.isfinite(float(deg)) else ""
        )
    except (TypeError, ValueError):
        suffix = ""
    return suffix


def _arrival_stage(label: str, action: str) -> tuple[str, str] | None:
    if "arrival confirm" in action:
        return "arriving", f"到達{label}確認中"
    if action == "WAIT_MAP_CONFIRMATION":
        return "map_confirmation", f"等待地圖定位確認（{label}）"
    return None
