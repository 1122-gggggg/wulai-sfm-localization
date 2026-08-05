#!/usr/bin/env bash
# Resolve the active local X/XWayland display after login or reboot.

configure_operator_display() {
  local dry_run="${1:-0}"
  local socket_dir="${SFM_X11_SOCKET_DIR:-/tmp/.X11-unix}"
  local candidate=""
  local socket=""

  if [[ -n "${DISPLAY:-}" ]]; then
    if [[ "$DISPLAY" != :* ]]; then
      return 0
    fi
    if [[ "$DISPLAY" =~ ^:([0-9]+)(\.[0-9]+)?$ ]] \
       && [[ -S "$socket_dir/X${BASH_REMATCH[1]}" ]]; then
      return 0
    fi
  fi

  if [[ -d "$socket_dir" ]]; then
    while IFS= read -r socket; do
      [[ -S "$socket" ]] || continue
      candidate=":${socket##*/X}"
      if command -v xdpyinfo >/dev/null 2>&1; then
        if DISPLAY="$candidate" xdpyinfo >/dev/null 2>&1; then
          break
        fi
        candidate=""
      else
        break
      fi
    done < <(find "$socket_dir" -maxdepth 1 -type s -name 'X[0-9]*' -print 2>/dev/null | sort -V)
  fi

  if [[ -z "$candidate" ]]; then
    if [[ "$dry_run" == "1" ]]; then
      candidate=":0"
    else
      echo "[display] ERROR: 找不到可用的本機 X/XWayland DISPLAY" >&2
      echo "[display] 請先登入桌面工作階段，或明確設定 DISPLAY" >&2
      return 1
    fi
  fi
  export DISPLAY="$candidate"

  if [[ -z "${XAUTHORITY:-}" ]]; then
    local runtime_dir="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    if [[ -f "$runtime_dir/gdm/Xauthority" ]]; then
      export XAUTHORITY="$runtime_dir/gdm/Xauthority"
    fi
  fi
  echo "[display] DISPLAY=$DISPLAY XAUTHORITY=${XAUTHORITY:-unset}"
}
