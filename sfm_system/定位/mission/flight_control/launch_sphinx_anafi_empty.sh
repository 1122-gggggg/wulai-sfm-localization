#!/usr/bin/env bash
set -euo pipefail

DRONE_FILE="${DRONE_FILE:-/opt/parrot-sphinx/usr/share/sphinx/drones/anafi.drone}"
FIRMWARE_URL="${FIRMWARE_URL:-https://firmware.parrot.com/Versions/anafi/pc/%23latest/images/anafi-pc.ext2.zip}"
UE_DELAY_S="${UE_DELAY_S:-8}"

sphinx "${DRONE_FILE}"::firmware="${FIRMWARE_URL}" &
SPHINX_PID=$!

cleanup() {
  kill "${UE_PID:-}" "${SPHINX_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

sleep "${UE_DELAY_S}"
parrot-ue4-empty &
UE_PID=$!

wait "${UE_PID}"
