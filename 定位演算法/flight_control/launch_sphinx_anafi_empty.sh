#!/usr/bin/env bash
set -euo pipefail

DRONE_FILE="${DRONE_FILE:-/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone}"
FIRMWARE_URL="${FIRMWARE_URL:-}"
UE4_BIN="${UE4_BIN:-/opt/parrot-ue4-empty/Empty/Binaries/Linux/UnrealApp-Linux-Shipping}"
UE4_MAP="${UE4_MAP:-Empty}"
UE4_USERDIR="${UE4_USERDIR:-${HOME}/.parrot-sphinx/unreal_sphinx}"
SPHINX_IP="10.202.0.1"
UE_DELAY_S="${UE_DELAY_S:-8}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-45}"
CLEANUP_TIMEOUT_S="${CLEANUP_TIMEOUT_S:-5}"
SPHINX_LOG="${SPHINX_LOG:-${TMPDIR:-/tmp}/sfm_sphinx_$$.log}"

SPHINX_PID=""
UE_PID=""

validate_firmware_url() {
  if [[ -z "${FIRMWARE_URL}" ]]; then
    echo "[error] FIRMWARE_URL must name an explicit ANAFI PC firmware revision; latest is forbidden." >&2
    return 1
  fi
  local selector="${FIRMWARE_URL,,}"
  if [[ "${selector}" == *"latest"* ]]; then
    echo "[error] FIRMWARE_URL uses the mutable latest selector; provide an explicit revision URL." >&2
    return 1
  fi
  if [[ "${selector}" != https://firmware.parrot.com/versions/anafi/pc/* ]]; then
    echo "[error] FIRMWARE_URL must be an HTTPS ANAFI PC image URL from firmware.parrot.com." >&2
    return 1
  fi
  local suffix="${selector#https://firmware.parrot.com/versions/anafi/pc/}"
  if [[ ! "${suffix}" =~ ^[^/?#]+/images/[^/?#]+\.zip$ ]]; then
    echo "[error] FIRMWARE_URL must include an explicit revision and firmware image path." >&2
    return 1
  fi
}

terminate_group() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
}

group_alive() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 -- "-${pid}" 2>/dev/null
}

cleanup() {
  trap - EXIT INT TERM
  terminate_group "${UE_PID}"
  terminate_group "${SPHINX_PID}"
  local deadline=$((SECONDS + CLEANUP_TIMEOUT_S))
  while (( SECONDS < deadline )); do
    if ! group_alive "${UE_PID}" && ! group_alive "${SPHINX_PID}"; then
      break
    fi
    sleep 0.1
  done
  if group_alive "${UE_PID}"; then
    echo "[cleanup] UE4 ignored TERM; sending KILL" >&2
    kill -KILL -- "-${UE_PID}" 2>/dev/null || true
  fi
  if group_alive "${SPHINX_PID}"; then
    echo "[cleanup] Sphinx ignored TERM; sending KILL" >&2
    kill -KILL -- "-${SPHINX_PID}" 2>/dev/null || true
  fi
  [[ -z "${UE_PID}" ]] || wait "${UE_PID}" 2>/dev/null || true
  [[ -z "${SPHINX_PID}" ]] || wait "${SPHINX_PID}" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

validate_firmware_url

for command in systemctl setsid sphinx ping grep tee; do
  command -v "${command}" >/dev/null || {
    echo "[error] required command not found: ${command}" >&2
    exit 1
  }
done
[[ -r "${DRONE_FILE}" ]] || { echo "[error] ANAFI drone file not readable: ${DRONE_FILE}" >&2; exit 1; }
[[ -x "${UE4_BIN}" ]] || { echo "[error] offscreen UE4 binary not executable: ${UE4_BIN}" >&2; exit 1; }

if [[ "${1:-}" == "--check" ]]; then
  systemctl is-active --quiet firmwared || {
    echo "[check] firmwared is not active; run: systemctl start firmwared" >&2
    exit 1
  }
  echo "[check] $(sphinx --version 2>&1 | head -n 1)"
  echo "[check] explicit firmware selector: ${FIRMWARE_URL}"
  echo "[check] exact firmware and Olympe versions are recorded by the convergence harness."
  echo "[check] Sphinx, ANAFI model, firmwared, and offscreen UE4 binary are available."
  exit 0
fi
if [[ $# -gt 0 ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

if ! systemctl is-active --quiet firmwared; then
  echo "[startup] firmwared is inactive; requesting systemd start..."
  systemctl start firmwared
fi
systemctl is-active --quiet firmwared || {
  echo "[error] firmwared did not become active" >&2
  exit 1
}

echo "[startup] Parrot Sphinx core with ANAFI PC firmware"
echo "[runtime] $(sphinx --version 2>&1 | head -n 1)"
echo "[runtime] explicit firmware selector: ${FIRMWARE_URL}"
echo "[runtime] Sphinx log: ${SPHINX_LOG}"
: > "${SPHINX_LOG}"
setsid sphinx "${DRONE_FILE}::firmware=${FIRMWARE_URL}" \
  > >(tee -a "${SPHINX_LOG}") 2>&1 &
SPHINX_PID=$!

sleep "${UE_DELAY_S}"
kill -0 "${SPHINX_PID}" 2>/dev/null || {
  echo "[error] Sphinx exited before UE4 startup" >&2
  exit 1
}

echo "[startup] known-working offscreen UE4 renderer"
setsid "${UE4_BIN}" "${UE4_MAP}" \
  -preferNvidia -nosound -userdir="${UE4_USERDIR}" -RenderOffScreen &
UE_PID=$!

deadline=$((SECONDS + READY_TIMEOUT_S))
while (( SECONDS < deadline )); do
  kill -0 "${SPHINX_PID}" 2>/dev/null || {
    echo "[error] Sphinx exited before the simulated ANAFI became ready" >&2
    exit 1
  }
  kill -0 "${UE_PID}" 2>/dev/null || {
    echo "[error] offscreen UE4 exited before the simulated ANAFI became ready" >&2
    exit 1
  }
  if grep -q "All drones dropped" "${SPHINX_LOG}"; then
    echo "[error] Sphinx reported 'All drones dropped'; retry the full stack." >&2
    exit 1
  fi
  if grep -q "All drones instantiated" "${SPHINX_LOG}" && \
      ping -c 1 -W 1 "${SPHINX_IP}" >/dev/null 2>&1; then
    echo "[ready] Sphinx reported 'All drones instantiated' and ${SPHINX_IP} responds."
    echo "[ready] Run the smoke test or convergence harness in another terminal."
    echo "[ready] If Sphinx prints 'All drones dropped', stop this launcher and retry the full stack."
    set +e
    wait -n "${SPHINX_PID}" "${UE_PID}"
    child_status=$?
    set -e
    [[ ${child_status} -ne 0 ]] || child_status=1
    echo "[stopped] a simulator child exited with status ${child_status}; cleaning up both process groups." >&2
    exit "${child_status}"
  fi
  sleep 1
done

echo "[error] ${SPHINX_IP} was not ready after ${READY_TIMEOUT_S}s." >&2
echo "[error] Check console output for 'All drones instantiated' versus 'All drones dropped', then retry." >&2
exit 1
