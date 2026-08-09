#!/usr/bin/env bash
# ANAFI live operator UI via SkyController 3 (Parrot white paper path).
#
# Architecture (field):
#   PC USB ──► SkyController 3 ──Wi‑Fi──► ANAFI
#   - Live stream (controller): HD 720p H.264  (white paper)
#   - PC sends PCMD when pilotingSource=Controller
#   - Esc / 手動 restores SkyController sticks
#   - Ctrl‑C / close window / kill terminal → zero PCMD + Landing + sticks
#
# Direct drone Wi‑Fi (192.168.42.1) is NOT used when SC already owns the link
# (ANAFI is single-controller).

set -euo pipefail
OI="$(cd "$(dirname "$0")" && pwd)"
# Workspace root: .../localization (contains 控制介面程式 / 定位演算法 / 地圖檔)
WORKSPACE_ROOT="$(cd "$OI/../.." && pwd)"
export SFM_WORKSPACE_ROOT="${SFM_WORKSPACE_ROOT:-$WORKSPACE_ROOT}"
PACKAGE_ROOT="$WORKSPACE_ROOT"
DRY_RUN="${SFM_LAUNCH_DRY_RUN:-0}"
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
  echo "[start] ERROR: SFM_LAUNCH_DRY_RUN must be 0 or 1" >&2
  exit 2
fi

verify_portable_package() {
  if [[ ! -f "$PACKAGE_ROOT/PORTABLE_PACKAGE.json" ]]; then
    if [[ ! -d "$PACKAGE_ROOT/.git" ]]; then
      echo "[start] ERROR: non-Git runtime is missing PORTABLE_PACKAGE.json" >&2
      exit 2
    fi
    return 0
  fi
  local verifier
  verifier="$(command -v python3.10 || command -v python3 || true)"
  if [[ -z "$verifier" ]]; then
    echo "[start] ERROR: no Python interpreter available for portable manifest verification" >&2
    exit 2
  fi
  if [[ ! -f "$PACKAGE_ROOT/tools/package_manifest.py" ]]; then
    echo "[start] ERROR: portable package is missing tools/package_manifest.py" >&2
    exit 2
  fi
  echo "[start] verifying portable package manifest ..."
  if ! "$verifier" "$PACKAGE_ROOT/tools/package_manifest.py" verify --root "$PACKAGE_ROOT"; then
    echo "[start] ERROR: portable package manifest verification failed" >&2
    exit 1
  fi
}

verify_portable_package

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法: ./start_anafi_live.sh [flight_operator_app.py 的真機選項]

此入口只會啟動 real-flight Olympe 介面，禁止傳入 --video 或
--interface simulated-stream。它可能連線 SkyController/ANAFI；離線檢查請使用：
  控制介面程式/影片模擬串流/選擇啟動.sh
EOF
  exit 0
fi
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
source "$OI/resolve_display.sh"
configure_operator_display "$DRY_RUN"
for arg in "$@"; do
  case "$arg" in
    --interface|--interface=*|--video|--video=*)
      echo "[start] ERROR: real-flight launcher rejects cross-interface argument: $arg" >&2
      echo "[start] Use 控制介面程式/影片模擬串流/啟動.sh for video files" >&2
      exit 2
      ;;
  esac
done
MAX_PERFORMANCE="${SFM_MAX_PERFORMANCE:-1}"
if [[ "$MAX_PERFORMANCE" != "0" && "$MAX_PERFORMANCE" != "1" ]]; then
  echo "[start] ERROR: SFM_MAX_PERFORMANCE must be 0 or 1" >&2
  exit 2
fi
if [[ "$MAX_PERFORMANCE" == "1" ]]; then
  CPU_THREADS="${SFM_CPU_THREADS:-4}"
  if [[ ! "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[start] ERROR: SFM_CPU_THREADS must be a positive integer" >&2
    exit 2
  fi
  # Four threads is the verified sustained-performance point on the deployment
  # laptop: higher defaults thermally throttle; two threads do not improve FPS.
  export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-$CPU_THREADS}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_THREADS}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_THREADS}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_THREADS}"
  echo "[start] sustained CPU thread budget: $CPU_THREADS"
fi

# UI interpreter resolution. VENV remains a compatibility override and denotes
# a Python executable, not a directory.
UI_PYTHON="${SFM_UI_PYTHON:-${VENV:-}}"
if [[ -z "$UI_PYTHON" ]]; then
  # Support both a package-local venv and a workspace venv one or two levels
  # above a transferred/nested package, without hard-coding a user's home.
  SEARCH_ROOT="$PACKAGE_ROOT"
  for _ in 1 2 3; do
    if [[ -x "$SEARCH_ROOT/.venv/bin/python" ]]; then
      UI_PYTHON="$SEARCH_ROOT/.venv/bin/python"
      break
    fi
    SEARCH_ROOT="$(dirname "$SEARCH_ROOT")"
  done
fi
if [[ -z "$UI_PYTHON" ]]; then
  UI_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$UI_PYTHON" ]]; then
  echo "[start] ERROR: no Python interpreter found; set SFM_UI_PYTHON" >&2
  exit 2
fi
if [[ "$UI_PYTHON" == */* ]]; then
  if [[ ! -x "$UI_PYTHON" ]]; then
    echo "[start] ERROR: Python is not executable: $UI_PYTHON" >&2
    exit 2
  fi
  UI_PYTHON="$(cd "$(dirname "$UI_PYTHON")" && pwd)/$(basename "$UI_PYTHON")"
else
  UI_PYTHON="$(command -v "$UI_PYTHON" || true)"
  if [[ -z "$UI_PYTHON" ]]; then
    echo "[start] ERROR: Python command not found" >&2
    exit 2
  fi
fi
if [[ -f "$PACKAGE_ROOT/PORTABLE_PACKAGE.json" ]]; then
  if [[ -L "$PACKAGE_ROOT/.venv" ]]; then
    echo "[start] ERROR: portable package-local .venv must not be a symlink" >&2
    exit 2
  fi
  case "$UI_PYTHON" in
    "$PACKAGE_ROOT/.venv/"*)
      PORTABLE_MARKER="$PACKAGE_ROOT/.venv/.sfm-portable-runtime"
      IDENTITY_PYTHON="$(command -v python3.10 || command -v python3 || true)"
      PORTABLE_IDENTITY="$("$IDENTITY_PYTHON" - "$PACKAGE_ROOT" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for name in ("PORTABLE_PACKAGE.json", "MANIFEST.tsv"):
    digest.update(name.encode("utf-8") + b"\0")
    digest.update((root / name).read_bytes())
print(digest.hexdigest())
PY
)"
      if [[ ! -f "$PORTABLE_MARKER" ]] \
         || [[ "$(<"$PORTABLE_MARKER")" != "$PORTABLE_IDENTITY" ]]; then
        echo "[start] ERROR: package-local .venv is not bound to this portable manifest" >&2
        exit 2
      fi
      ;;
  esac
fi
if [[ "$DRY_RUN" != "1" ]]; then
  UI_VENV_CFG="$(dirname "$(dirname "$UI_PYTHON")")/pyvenv.cfg"
  if [[ -f "$UI_VENV_CFG" ]] \
     && grep -qi '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "$UI_VENV_CFG"; then
    echo "[start] ERROR: real-flight refuses include-system-site-packages=true: $UI_VENV_CFG" >&2
    echo "[start] Rebuild a clean CPython 3.10 venv with tools/install_runtime.sh" >&2
    exit 2
  fi
fi
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

target_reachable() {
  local target="$1"
  if command -v ping >/dev/null 2>&1 \
     && ping -c 1 -W 1 "$target" >/dev/null 2>&1; then
    return 0
  fi
  # Some ANAFI firmware/network combinations answer HTTP/ARSDK while dropping
  # ICMP echo. An HTTP response (including a non-2xx status) proves reachability
  # without sending any flight command.
  if command -v curl >/dev/null 2>&1 \
     && curl --silent --show-error --output /dev/null \
       --connect-timeout 1 --max-time 2 "http://$target/"; then
    return 0
  fi
  return 1
}

# Modes:
#   SC USB (default):  IP=192.168.53.1 CTRL=skycontroller3
#   Direct drone WiFi: IP=192.168.42.1 CTRL=drone
#     example: IP=192.168.42.1 CTRL=drone ./start_anafi_live.sh
# Auto: if drone WiFi is up and SC is not, prefer direct. Dry-run never probes
# the network and uses the field-safe SkyController defaults.
if [[ -z "${IP:-}" && -z "${CTRL:-}" ]]; then
  if [[ "$DRY_RUN" == "1" ]]; then
    IP=192.168.53.1
    CTRL=skycontroller3
  elif target_reachable 192.168.42.1 \
       && ! target_reachable 192.168.53.1; then
    IP=192.168.42.1
    CTRL=drone
  else
    IP=192.168.53.1
    CTRL=skycontroller3
  fi
elif [[ -n "${IP:-}" && -z "${CTRL:-}" ]]; then
  case "$IP" in
    192.168.42.*) CTRL=drone ;;
    192.168.53.*) CTRL=skycontroller3 ;;
    *)
      echo "[start] ERROR: cannot infer controller for IP=$IP; set CTRL" >&2
      exit 2
      ;;
  esac
elif [[ -z "${IP:-}" && -n "${CTRL:-}" ]]; then
  case "$CTRL" in
    drone|anafi) IP=192.168.42.1 ;;
    sky*) IP=192.168.53.1 ;;
    *)
      echo "[start] ERROR: cannot infer IP for CTRL=$CTRL; set IP" >&2
      exit 2
      ;;
  esac
fi
IP="${IP:-192.168.53.1}"
CTRL="${CTRL:-skycontroller3}"

if [[ "$IP" == 192.168.42.* && "$CTRL" == sky* ]] \
   || [[ "$IP" == 192.168.53.* && ("$CTRL" == "drone" || "$CTRL" == "anafi") ]]; then
  echo "[start] ERROR: inconsistent IP/controller pair: $IP + $CTRL" >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  echo "[start] checking target at $IP (controller=$CTRL) ..."
fi
if [[ "$DRY_RUN" != "1" ]] && ! target_reachable "$IP"; then
  echo "[start] ERROR: cannot reach $IP"
  echo "  SkyController path:"
  echo "    1) Power on SC3 + ANAFI, USB PC↔SC, wait for drone link"
  echo "    2) ip a should show 192.168.53.x"
  echo "  Direct drone WiFi path:"
  echo "    1) Connect laptop to ANAFI-XXXX WiFi"
  echo "    2) IP=192.168.42.1 CTRL=drone $0"
  exit 1
fi

if [[ "$CTRL" == *sky* || "$IP" == 192.168.53.* ]]; then
  echo "[start] LIVE Operator UI (ANAFI via SkyController, 720p + PCMD)"
  echo "  safety pilot: hold sticks; Esc = take over; close/Ctrl‑C = land"
else
  echo "[start] LIVE Operator UI (ANAFI DIRECT WiFi $IP, 720p + PCMD)"
  echo "  NO SkyController — laptop is sole controller"
  echo "  Esc freezes PC PCMD; 恢復電腦控制 resumes; close/Ctrl‑C = land"
fi
# Optional env:
#   LOC_BENCH=1        -> auto 開始定位 + no boot-lock (BOOT_INIT/MegaLoc path; NO takeoff)
#   LOC_BENCH_TRACK=1  -> same + force TRACK path each frame (no MegaLoc; in-flight FPS)
#   LOCAL_TOPK=N       -> explicit TRACK local_topk override (0/profile default)
#   SFM_SITE_PROFILE=  -> required site profile JSON unless passed on the CLI
#   SFM_CPU_THREADS=N  -> override the verified four-thread sustained-performance budget
# Extra args after script are passed through (e.g. --no-live-localize).
EXTRA=()
# Never choose a field site implicitly. The app rejects --live without an
# explicit atomic profile, preventing a stale map/route/bundle combination.
SITE_PROFILE="${SFM_SITE_PROFILE:-}"
if [[ -n "$SITE_PROFILE" ]]; then
  EXTRA+=(--site-profile "$SITE_PROFILE")
  echo "[start] site-profile: $SITE_PROFILE"
fi
# Let the selected site's production profile choose TRACK top-k by default.
LOCAL_TOPK="${LOCAL_TOPK:-0}"
if [[ -n "$LOCAL_TOPK" && "$LOCAL_TOPK" != "0" ]]; then
  EXTRA+=(--local-topk "$LOCAL_TOPK")
  echo "[start] local_topk=$LOCAL_TOPK"
fi
if [[ "${LOC_BENCH:-0}" == "1" || "${LOC_BENCH_TRACK:-0}" == "1" ]]; then
  EXTRA+=(--auto-inspect --boot-lock-ms 0)
  echo "[start] LOC_BENCH: auto-inspect localization pipeline (no takeoff)"
fi
if [[ "${LOC_BENCH_TRACK:-0}" == "1" ]]; then
  EXTRA+=(--loc-force-track-bench)
  echo "[start] LOC_BENCH_TRACK=1: force TRACK path (skip MegaLoc/BOOT_INIT)"
fi
# YOLO is not part of this flight localization UI (default off).
# Optional: LOC_EVERY_N=2 to submit every 2nd stream frame to the localizer.
if [[ -n "${LOC_EVERY_N:-}" ]]; then
  EXTRA+=(--loc-every-n-frames "${LOC_EVERY_N}")
fi
# The distance geofence is deliberately NOT passed as a flag here. --distance-geofence
# defaults to env_bool("SFM_DISTANCE_GEOFENCE", False) in flight_operator_app, so
# leaving it off the command line keeps the same OFF default while letting
# SFM_DISTANCE_GEOFENCE=1 actually turn it on -- an explicit flag beat the env
# every time, which made the switch the --help text advertises unreachable from
# this launcher. Pass --distance-geofence after the script name to force it on.
if [[ -n "${SFM_DISTANCE_GEOFENCE:-}" ]]; then
  echo "[start] SFM_DISTANCE_GEOFENCE=${SFM_DISTANCE_GEOFENCE} (NoFlyOverMaxDistance)"
fi
CMD_PREFIX=()
if [[ "$MAX_PERFORMANCE" == "1" ]]; then
  GAMEMODE_RUN="$(command -v gamemoderun || true)"
  if [[ -n "$GAMEMODE_RUN" ]]; then
    CMD_PREFIX+=("$GAMEMODE_RUN")
  elif [[ "$DRY_RUN" != "1" ]]; then
    echo "[start] WARNING: gamemoderun is unavailable; continuing without it" >&2
  fi
fi
CMD=(
  "${CMD_PREFIX[@]}"
  "$UI_PYTHON" -u "$OI/flight_operator_app.py" --interface real-flight
  --ip "$IP" --controller "$CTRL"
  --no-live-detect
  --max-altitude-m "${SFM_MAX_ALTITUDE_M:-50}"
  --max-distance-m "${SFM_MAX_DISTANCE_M:-100}"
  --rth-min-altitude-m "${SFM_RTH_MIN_ALTITUDE_M:-20.0}"
  --stream-loss-grace-s "${SFM_STREAM_LOSS_GRACE_S:-10.0}"
  --nudge-pct "${NUDGE_PCT:-8}"
  --nudge-pulse-s "${NUDGE_PULSE_S:-0.20}"
  "${EXTRA[@]}"
  "$@"
)
if [[ "$DRY_RUN" == "1" ]]; then
  printf '[start] dry-run command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

# powerprofilesctl and this GNOME key change session state. Keep both changes
# scoped to this launcher: remember the old values and restore them after the
# UI exits, including error and signal exits. GameMode already has this
# process-lifetime behavior.
ORIGINAL_POWER_PROFILE=""
ORIGINAL_LOW_BATTERY_SAVER=""
POWER_PROFILE_CHANGED=0
LOW_BATTERY_SAVER_CHANGED=0

restore_performance_settings() {
  local app_status="$1"
  trap - EXIT
  if [[ "$LOW_BATTERY_SAVER_CHANGED" == "1" ]]; then
    if ! gsettings set org.gnome.settings-daemon.plugins.power \
      power-saver-profile-on-low-battery "$ORIGINAL_LOW_BATTERY_SAVER"; then
      echo "[start] WARNING: could not restore the low-battery power-saver setting" >&2
    fi
  fi
  if [[ "$POWER_PROFILE_CHANGED" == "1" ]]; then
    if ! powerprofilesctl set "$ORIGINAL_POWER_PROFILE"; then
      echo "[start] WARNING: could not restore power profile '$ORIGINAL_POWER_PROFILE'" >&2
    fi
  fi
  exit "$app_status"
}
trap 'restore_performance_settings $?' EXIT

if [[ "$MAX_PERFORMANCE" == "1" ]]; then
  if command -v powerprofilesctl >/dev/null 2>&1; then
    if ORIGINAL_POWER_PROFILE="$(powerprofilesctl get 2>/dev/null)" \
       && [[ -n "$ORIGINAL_POWER_PROFILE" ]]; then
      if [[ "$ORIGINAL_POWER_PROFILE" != "performance" ]]; then
        if powerprofilesctl set performance; then
          POWER_PROFILE_CHANGED=1
        else
          echo "[start] WARNING: could not select the performance power profile; continuing without it" >&2
        fi
      fi
      ACTIVE_POWER_PROFILE="$(powerprofilesctl get 2>/dev/null || true)"
      if [[ "$ACTIVE_POWER_PROFILE" == "performance" ]]; then
        if [[ "$ORIGINAL_POWER_PROFILE" != "performance" ]]; then
          POWER_PROFILE_CHANGED=1
        fi
        echo "[start] active power profile: performance"
      else
        echo "[start] WARNING: active power profile is '${ACTIVE_POWER_PROFILE:-unknown}', not performance" >&2
      fi
    else
      ORIGINAL_POWER_PROFILE=""
      echo "[start] WARNING: could not read the current power profile; left it unchanged" >&2
    fi
  else
    echo "[start] WARNING: powerprofilesctl is unavailable; continuing without it" >&2
  fi
  if command -v gsettings >/dev/null 2>&1; then
    ORIGINAL_LOW_BATTERY_SAVER="$(
      gsettings get org.gnome.settings-daemon.plugins.power \
        power-saver-profile-on-low-battery 2>/dev/null || true
    )"
    if [[ "$ORIGINAL_LOW_BATTERY_SAVER" == "true" ]]; then
      if gsettings set org.gnome.settings-daemon.plugins.power \
        power-saver-profile-on-low-battery false; then
        LOW_BATTERY_SAVER_CHANGED=1
      else
        echo "[start] WARNING: could not disable automatic low-battery power saver; continuing without it" >&2
      fi
    elif [[ "$ORIGINAL_LOW_BATTERY_SAVER" != "false" ]]; then
      ORIGINAL_LOW_BATTERY_SAVER=""
      echo "[start] WARNING: could not read the low-battery power-saver setting; left it unchanged" >&2
    fi
    ACTIVE_LOW_BATTERY_SAVER="$(
      gsettings get org.gnome.settings-daemon.plugins.power \
        power-saver-profile-on-low-battery 2>/dev/null || true
    )"
    if [[ "$ACTIVE_LOW_BATTERY_SAVER" == "false" ]]; then
      if [[ "$ORIGINAL_LOW_BATTERY_SAVER" == "true" ]]; then
        LOW_BATTERY_SAVER_CHANGED=1
      fi
      echo "[start] automatic low-battery power saver: disabled for this run"
    else
      echo "[start] WARNING: automatic low-battery power saver is not confirmed disabled" >&2
    fi
  else
    echo "[start] WARNING: gsettings is unavailable; continuing without it" >&2
  fi
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "[start] WARNING: DISPLAY/WAYLAND_DISPLAY is unset; the UI may not open" >&2
fi
APP_PID=""
forward_ui_signal() {
  local signal_name="$1"
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill -s "$signal_name" "$APP_PID" 2>/dev/null || true
  fi
}
trap 'forward_ui_signal INT' INT
trap 'forward_ui_signal TERM' TERM
trap 'forward_ui_signal HUP' HUP

"${CMD[@]}" &
APP_PID=$!
while true; do
  if wait "$APP_PID"; then
    APP_STATUS=0
  else
    APP_STATUS=$?
  fi
  # wait can be interrupted by this shell's signal trap before the UI has
  # finished its own cleanup. Keep waiting so settings are not restored early.
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    break
  fi
done
exit "$APP_STATUS"
