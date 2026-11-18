"""Operator-facing HUD / inventory / calibration text formatters."""
from __future__ import annotations

import math
import time


def format_map_scale(snapshot: dict) -> str:
    """Show a candidate distinctly from a usable map-wide metric scale."""
    candidate = snapshot.get("candidate_m_per_unit")
    reliable = bool(snapshot.get("reliable"))
    value = snapshot.get("scale_m_per_unit") if reliable else candidate
    estimate = "-" if value is None else f"{float(value):.2f}"
    reason = {
        "converged": "可用",
        "insufficient_segments": "累積中",
        "insufficient_displacement": "位移不足",
        "scale_iqr_unstable": "分散未收斂",
        "scale_drift_unstable": "前後尺度不一致",
        "scale_stale": "證據過期",
        "direction_mismatch": "位移方向不一致",
        "scale_outlier": "新尺度不一致",
    }.get(snapshot.get("reason"), "尚不可用")
    label = "尺度" if reliable else "尺度候選"
    spread = snapshot.get("relative_iqr")
    spread_text = "" if spread is None else f"｜IQR {float(spread):.0%}"
    return (
        f"全場{label} {estimate} m/u（{reason}）"
        f"｜{snapshot.get('scale_segments', 0)}段／"
        f"{float(snapshot.get('cumulative_distance_m') or 0.):.1f} m{spread_text}"
    )


def _telemetry_text(value: object) -> str:
    if value is None:
        return "?"
    text = str(getattr(value, "name", value)).rsplit(".", 1)[-1].strip()
    return text or "?"


def _telemetry_number(value: object, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "?"
    if not math.isfinite(number):
        return "?"
    return f"{number:.{digits}f}"


def _firmware_limit_text(value: object, suffix: str, digits: int = 1) -> str:
    if value is None:
        return "?"
    return f"{float(value):.{digits}f}{suffix}"


def _distance_geofence_text(state: object) -> str:
    enabled = getattr(state, "distance_geofence_enabled", None)
    guard_active = getattr(state, "distance_guard_active", None)
    if enabled is None:
        return "圍欄狀態未知"
    if not enabled:
        return "圍欄關閉（不阻擋起飛）"
    if guard_active is False:
        return "圍欄 ON 但未生效（無 GPS/Home；不阻擋起飛）"
    return "圍欄 ON"


def _active_anafi_incident(state: object, *, link_ok: bool) -> str:
    incident = str(getattr(state, "active_incident", "") or "")
    if not incident and not link_ok:
        return "CONTROL LINK LOST"
    stream = str(getattr(state, "stream", "")).upper()
    if not incident and stream in {"LOST", "DECODE_ERROR_HOLD"}:
        return stream
    return incident


def gps_operator_message(state: object, *, live: bool) -> tuple[str, str]:
    """Return a concise, non-blocking GPS status for the flight header."""
    if not live:
        return "GPS：模擬", "neutral"
    fixed = getattr(state, "gps_fixed", None)
    if fixed is True:
        satellites = getattr(state, "gps_satellites", None)
        suffix = "" if satellites is None else f"（{int(satellites)} 顆）"
        return f"GPS：已定位{suffix}", "good"
    if fixed is False:
        return "GPS：未定位｜手動可起飛，自動先懸停", "warning"
    return "GPS：等待讀回｜不阻擋手動起飛", "warning"


_INVENTORY_REASON_ZH = (
    ("hardware inventory not read", "硬體清單尚未讀回"),
    ("olympe version", "Olympe 版本尚無核准紀錄"),
    ("aircraft firmware", "飛機韌體版本尚無核准紀錄"),
    ("controller firmware", "控制器韌體版本尚無核准紀錄"),
    (
        "not confirmed skycontroller 3",
        "控制器已連線，但尚未從韌體或 USB HID 確認為 SkyController 3",
    ),
    ("formal flight requires skycontroller 3", "真機飛行需使用 SkyController 3"),
    (
        "connected aircraft is not an approved anafi",
        "已連線飛機尚未確認為核准的 ANAFI 機型",
    ),
    ("aircraft serial unavailable", "尚未讀到飛機序號"),
    ("controller serial unavailable", "尚未讀到控制器序號"),
    ("lost-link auto-land policy", "失聯後自動降落策略尚未讀回確認"),
)


def _inventory_reason_zh(reason: object) -> str:
    """Translate one backend inventory blocker without hiding its meaning."""
    text = str(reason or "").strip()
    lower = text.lower()
    if not text or lower == "ready":
        return ""
    for fragment, translated in _INVENTORY_REASON_ZH:
        if fragment in lower:
            return translated
    return text


def inventory_ui_status(backend: object) -> dict[str, tuple[str, str]]:
    """Split hardware/version/lost-link readiness into independent UI cards."""
    if not bool(getattr(backend, "is_live", False)):
        return {
            "hardware": ("good", "模擬硬體"),
            "version": ("good", "模擬版本"),
            "lost_link": ("good", "模擬失聯策略"),
        }

    inventory = getattr(backend, "connection_inventory", None)
    if not isinstance(inventory, dict) or not inventory:
        waiting = ("waiting", "等待讀回")
        return {"hardware": waiting, "version": waiting, "lost_link": waiting}

    raw_reasons = inventory.get("block_reasons", ())
    if not isinstance(raw_reasons, (list, tuple)):
        raw_reasons = (raw_reasons,)
    reasons = [str(reason) for reason in raw_reasons if str(reason).strip()]
    hardware_errors = [
        reason for reason in reasons
        if any(token in reason.lower() for token in (
            "aircraft serial", "controller serial", "approved anafi",
            "confirmed skycontroller", "requires skycontroller",
        ))
    ]
    version_errors = [
        reason for reason in reasons
        if any(token in reason.lower() for token in (
            "olympe version", "aircraft firmware", "controller firmware",
        ))
    ]
    lost_link_errors = [
        reason for reason in reasons
        if "lost-link" in reason.lower() or "rth inventory" in reason.lower()
    ]
    classified = set(hardware_errors + version_errors + lost_link_errors)
    hardware_errors.extend(reason for reason in reasons if reason not in classified)

    aircraft = inventory.get("aircraft") or {}
    controller = inventory.get("controller") or {}
    runtime = inventory.get("runtime") or {}
    lost_link = inventory.get("lost_link") or {}
    profile = getattr(backend, "aircraft_profile", None)
    aircraft_name = (
        aircraft.get("name") or aircraft.get("model")
        or getattr(profile, "model", None) or "機型未知"
    )
    controller_name = (
        controller.get("variant") or getattr(profile, "controller_name", None)
        or "控制器未知"
    )
    sdk_name = getattr(profile, "sdk_name", None) or "SDK"
    sdk_version_key = getattr(profile, "sdk_version_key", None)
    sdk_version = runtime.get(sdk_version_key) if sdk_version_key else None
    hardware_text = f"{aircraft_name} / {controller_name}"
    version_text = (
        f"{sdk_name} {sdk_version or '?'} · "
        f"飛機 {aircraft.get('software') or '?'} · "
        f"控制器 {controller.get('software') or '?'}"
    )
    fallback = str(lost_link.get("fallback") or "")
    lost_link_confirmed = lost_link.get("policy_confirmed") is True
    lost_link_text = (
        "已確認：無 Home 時原地降落"
        if fallback == "land_in_place"
        else "已確認：返航後降落"
        if fallback == "return_home_then_land"
        else "等待策略讀回"
    )
    return {
        "hardware": (
            "blocked" if hardware_errors else "good",
            _inventory_reason_zh(hardware_errors[0]) if hardware_errors else hardware_text,
        ),
        "version": (
            "blocked" if version_errors else "good",
            _inventory_reason_zh(version_errors[0]) if version_errors else version_text,
        ),
        "lost_link": (
            "blocked" if lost_link_errors or not lost_link_confirmed else "good",
            _inventory_reason_zh(lost_link_errors[0])
            if lost_link_errors
            else lost_link_text if lost_link_confirmed
            else "失聯後自動降落策略尚未讀回確認",
        ),
    }


def inventory_block_summary(backend: object) -> str:
    """Return the first actionable Chinese inventory blocker for preflight."""
    labels = {"hardware": "硬體", "version": "版本", "lost_link": "失聯策略"}
    statuses = inventory_ui_status(backend)
    for key in ("hardware", "version", "lost_link"):
        state, text = statuses[key]
        if state in {"blocked", "waiting"}:
            return f"{labels[key]}：{text}"
    raw = str(getattr(backend, "_inventory_block_reason", "") or "")
    return _inventory_reason_zh(raw) or "硬體清單尚未通過"


def format_olympe_telemetry(
    state: object,
    *,
    now_mono_ns: int | None = None,
) -> dict[str, str]:
    """Format safety-relevant values read from Olympe's local event cache."""
    roll = math.degrees(float(getattr(state, "att_roll", 0.0) or 0.0))
    pitch = math.degrees(float(getattr(state, "att_pitch", 0.0) or 0.0))
    yaw = math.degrees(float(getattr(state, "att_yaw", 0.0) or 0.0))

    gps_fixed = getattr(state, "gps_fixed", None)
    gps_fix_text = "?" if gps_fixed is None else ("FIX" if gps_fixed else "NO FIX")
    satellites = getattr(state, "gps_satellites", None)
    satellites_text = "?" if satellites is None else str(int(satellites))
    if gps_fixed is True:
        latitude = _telemetry_number(getattr(state, "gps_latitude_deg", None), digits=6)
        longitude = _telemetry_number(getattr(state, "gps_longitude_deg", None), digits=6)
        gps_altitude = _telemetry_number(getattr(state, "gps_altitude_m", None))
        accuracy_lat = _telemetry_number(
            getattr(state, "gps_latitude_accuracy_m", None), digits=1
        )
        accuracy_lon = _telemetry_number(
            getattr(state, "gps_longitude_accuracy_m", None), digits=1
        )
        accuracy_alt = _telemetry_number(
            getattr(state, "gps_altitude_accuracy_m", None), digits=1
        )
    else:
        # ANAFI reports large sentinel-like numbers while no fix is available.
        # Showing them as a real position/altitude misleads the operator.
        latitude = longitude = gps_altitude = "—"
        accuracy_lat = accuracy_lon = accuracy_alt = "—"

    warnings = []
    if getattr(state, "hover_no_gps_too_dark", False):
        warnings.append("無 GPS 且太暗")
    if getattr(state, "hover_no_gps_too_high", False):
        warnings.append("無 GPS 且過高")
    warning_text = "、".join(warnings) if warnings else "無"

    raw_link_quality = getattr(state, "link_signal_quality_raw", None)
    if raw_link_quality is None:
        link_quality_text = "?"
    else:
        raw_link_quality = int(raw_link_quality)
        flags = []
        if raw_link_quality & 0x40:
            flags.append("疑似 4G 干擾")
        if raw_link_quality & 0x80:
            flags.append("外部干擾")
        flag_text = f" ({'、'.join(flags)})" if flags else ""
        link_quality_text = f"{raw_link_quality & 0x0F}/5{flag_text}"

    rth_text = (
        f"RTH {_telemetry_text(getattr(state, 'navigate_home_state', None))}/"
        f"{_telemetry_text(getattr(state, 'navigate_home_reason', None))}"
    )
    attitude_text = (
        f"飛控融合姿態 roll {roll:+.1f}° | pitch {pitch:+.1f}° | "
        f"yaw {yaw:+.1f}°"
    )
    velocity_text = (
        "三軸速度 "
        f"N {_telemetry_number(getattr(state, 'speed_north_mps', None))} | "
        f"E {_telemetry_number(getattr(state, 'speed_east_mps', None))} | "
        f"D {_telemetry_number(getattr(state, 'speed_down_mps', None))} m/s"
    )
    altitude_agl_text = (
        "飛控高度(相對起飛點) "
        f"{_telemetry_number(getattr(state, 'drone_altitude_m', None))} m | "
        f"AGL {_telemetry_number(getattr(state, 'agl_altitude_m', None))} m"
    )
    gps_text = (
        f"GPS {gps_fix_text} | 衛星 {satellites_text} | "
        f"lat {latitude}, lon {longitude} | 1σ(m) "
        f"lat {accuracy_lat} lon {accuracy_lon} alt {accuracy_alt}"
    )

    sensor_states = getattr(state, "sensor_states", {}) or {}
    sensor_text = "、".join(
        f"{name}:{'OK' if ok else 'FAULT'}"
        for name, ok in sorted(sensor_states.items(), key=lambda item: item[0].lower())
    ) or "?"

    telemetry_stamp = getattr(state, "telemetry_read_mono_ns", None)
    if telemetry_stamp is None:
        cache_age_text = "?"
    else:
        now_ns = time.monotonic_ns() if now_mono_ns is None else int(now_mono_ns)
        cache_age_text = f"{max(0.0, (now_ns - int(telemetry_stamp)) / 1_000_000.0):.0f} ms"

    return {
        "state": (
            f"飛行 {_telemetry_text(getattr(state, 'flight_state', None))} | "
            f"警示 {_telemetry_text(getattr(state, 'alert_state', None))} | "
            f"航向 {_telemetry_text(getattr(state, 'heading_state', None))} | "
            f"{rth_text}"
        ),
        "rth": rth_text,
        "attitude": attitude_text,
        "speed": (
            f"{velocity_text} | "
            f"水平 {_telemetry_number(getattr(state, 'ground_speed_mps', None))} m/s"
        ),
        "velocity": velocity_text,
        "altitude": (
            f"{altitude_agl_text} | "
            f"GPS {gps_altitude} m"
        ),
        "altitude_agl": altitude_agl_text,
        "gps": gps_text,
        "link_quality": f"連接品質 {link_quality_text}",
        "environment": (
            f"風 {_telemetry_text(getattr(state, 'wind_state', None))} | "
            f"震動 {_telemetry_text(getattr(state, 'vibration_state', None))} | "
            f"懸停警告 {warning_text} | RSSI "
            f"{_telemetry_number(getattr(state, 'wifi_rssi_dbm', None), digits=0)} dBm | "
            f"鏈路品質 {link_quality_text}"
        ),
        "sensors": f"感測器健康 {sensor_text} | Olympe cache age {cache_age_text}",
    }


def format_device_telemetry(state: object) -> tuple[str, ...]:
    """Compact read-only device details; missing telemetry is never 'normal'."""
    def event(group: str, name: str) -> dict:
        values = getattr(state, group, {}) or {}
        result = values.get(name)
        return result if isinstance(result, dict) else {}

    def text(value) -> str:
        return "未知" if value is None else str(value)

    def number(value, digits=1) -> str:
        result = _telemetry_number(value, digits=digits)
        return "未知" if result == "?" else result

    def flags(value) -> str:
        return "未知" if value is None else (str(value) or "無")

    battery = getattr(state, "battery_info", {}) or {}
    voltage = event("battery_info", "voltage").get("voltage")
    voltage_v = None if voltage is None else float(voltage) / 1000.0
    controller = event("battery_info", "BatteryChanged").get("percent")
    controller_text = "充電中" if controller == 255 else f"{number(controller, 0)}%"
    alerts = battery.get("alert")
    alert_text = "未知"
    if isinstance(alerts, dict) and alerts:
        entries = [alerts] if "alert" in alerts else list(alerts.values())
        active = [
            f"{text(entry.get('alert'))}:{text(entry.get('level'))}"
            for entry in entries if isinstance(entry, dict) and entry.get("level") != "none"
        ]
        alert_text = "、".join(active) if active else "無"

    exposure = event("camera_info", "exposure")
    shutter = exposure.get("shutter_speed")
    shutter_text = (
        "未知" if shutter is None
        else str(shutter).removeprefix("shutter_").replace("_over_", "/") + " s"
    )
    iso = exposure.get("iso_sensitivity")
    iso_text = "未知" if iso is None else str(iso).removeprefix("iso_")
    white_balance = event("camera_info", "white_balance")
    wb_text = text(white_balance.get("mode"))
    if white_balance.get("mode") == "custom":
        wb_text += "/" + text(white_balance.get("temperature"))
    recording = event("camera_info", "recording_mode")
    bitrate = recording.get("bitrate")
    mbps = None if not bitrate else float(bitrate) / 1_000_000.0
    framerate = recording.get("framerate")
    fps_text = "未知" if framerate is None else str(framerate).removeprefix("fps_")
    zoom = event("camera_info", "zoom_info")

    def angles(prefix: str) -> str:
        return "/".join(
            number(getattr(state, f"gimbal_{prefix}{axis}_deg", None))
            for axis in ("roll", "pitch", "yaw")
        )

    return (
        f"電池 {number(getattr(state, 'battery_pct', None), 0)}% | "
        f"{number(voltage_v, 2)} V | "
        f"健康 {number(event('battery_info', 'health').get('state_of_health'), 0)}% | "
        f"循環 {number(event('battery_info', 'cycle_count').get('count'), 0)} | "
        f"遙控器 {controller_text}",
        f"電池警告 {alert_text} | "
        f"遙控器電池 {text(event('battery_info', 'BatteryState').get('state'))}",
        f"相機 {text(event('camera_info', 'camera_mode').get('mode'))} | "
        f"曝光 {text(event('camera_info', 'exposure_settings').get('mode'))} "
        f"{shutter_text} ISO {iso_text} | 白平衡 {wb_text}",
        f"機載錄影 {text(event('camera_info', 'recording_state').get('state'))} | "
        f"{text(recording.get('resolution'))} {fps_text} fps {number(mbps)} Mb/s | "
        f"無損變焦上限 {number(zoom.get('high_quality_maximum_level'))}x",
        f"雲台絕對 R/P/Y {angles('')}° | 相對 R/P/Y {angles('relative_')}°",
        f"雲台校正 {text(event('gimbal_info', 'calibration_state').get('state'))} | "
        f"穩定 {text(event('gimbal_info', 'stabilization_state').get('state'))} | "
        f"鎖軸 {flags(event('gimbal_info', 'axis_lock_state').get('locked'))} | "
        f"錯誤 {flags(event('gimbal_info', 'alert').get('error'))}",
    )


#: Which rotation the operator must perform for each firmware-requested axis.
#: ``view`` selects the matching aircraft projection in the renderer.
MAGNETOMETER_AXIS_GUIDE = {
    "x": ("X/roll", "top", "沿機頭—機尾軸滾轉（像轉烤肉串）"),
    "y": ("Y/pitch", "side", "沿左右翼尖軸翻轉（機頭往上翻過頂）"),
    "z": ("Z/yaw", "yaw", "機身保持水平、原地自轉（像轉盤）"),
}


def magnetometer_axis_guide(axis_raw: object) -> tuple[str, str, str] | None:
    """Map a firmware axis readback to (label, view, instruction).

    Returns None when no axis is being requested, so the caller can hide the
    diagram rather than show a stale rotation the operator should not perform.
    """
    key = str(axis_raw or "unknown").rsplit(".", 1)[-1].replace("_", "").lower()
    if key.endswith("axis"):
        key = key[:-4]
    return MAGNETOMETER_AXIS_GUIDE.get(key)


def format_magnetometer_calibration(state: object) -> dict[str, str]:
    """Format aircraft/controller firmware calibration without guessing state."""
    required = getattr(state, "drone_magnetometer_required", None)
    requirement = {0: "有效", 1: "必需", 2: "建議"}.get(required, "未知")
    started_state = getattr(state, "drone_magnetometer_started", None)
    started = started_state is True
    failed = getattr(state, "drone_magnetometer_failed", None) is True
    axis_results = (
        getattr(state, "drone_magnetometer_x_done", None),
        getattr(state, "drone_magnetometer_y_done", None),
        getattr(state, "drone_magnetometer_z_done", None),
    )
    if failed:
        axis_results = (False, False, False)
    axis_raw = str(getattr(state, "drone_magnetometer_axis", "unknown") or "unknown")
    axis_key = axis_raw.rsplit(".", 1)[-1].replace("_", "").lower()
    axis = {
        "x": "X/roll",
        "xaxis": "X/roll",
        "y": "Y/pitch",
        "yaxis": "Y/pitch",
        "z": "Z/yaw",
        "zaxis": "Z/yaw",
        "none": "完成／無",
    }.get(axis_key, "未知")

    def done(value: object) -> str:
        return "✓" if value is True else ("·" if value is False else "?")

    if failed:
        progress = "失敗，請移離金屬／磁場干擾後重試"
        result = "最新讀回校正結果：FAIL（韌體回報失敗），不可沿用"
    elif started:
        progress = f"進行中：目前 {axis}"
        result = "最新讀回校正結果：進行中；完成後會自動更新"
    elif started_state is not False:
        progress = "等待韌體校正進行狀態讀回"
        result = "校正結果：未知，尚不能判定是否可沿用"
    elif required == 0:
        progress = "待命"
        detail = "X/Y/Z 已完成" if all(value is True for value in axis_results) else "韌體回報有效"
        result = f"PASS：{detail}，可沿用"
    elif required == 2:
        progress = "待命，可選擇重新校正"
        result = "機上既有校正結果：可沿用；韌體建議重新校正"
    elif required == 1:
        progress = "等待使用者開始"
        result = "機上既有校正結果：不可沿用；必須完成重新校正"
    else:
        progress = "等待韌體狀態讀回"
        result = "校正結果：未知，尚不能判定是否可沿用"
    drone = (
        f"飛機羅盤：{requirement} | {progress} | "
        f"X{done(axis_results[0])} Y{done(axis_results[1])} Z{done(axis_results[2])}\n"
        f"{result}"
    )

    controller_raw = str(
        getattr(state, "skycontroller_magnetometer_state", "unknown") or "unknown"
    )
    controller_key = controller_raw.rsplit(".", 1)[-1].replace("_", "").lower()
    controller_label = {
        "notapplicable": "不適用（非 SkyController 連線）",
        "notcalibrated": "需要校正",
        "calibratingx": "進行中：X 軸",
        "calibratingy": "進行中：Y 軸",
        "calibratingz": "進行中：Z 軸",
        "calibrated": "已校正",
    }.get(controller_key, "未知")
    return {
        "drone": drone,
        "controller": f"SkyController 羅盤：{controller_label}",
    }
