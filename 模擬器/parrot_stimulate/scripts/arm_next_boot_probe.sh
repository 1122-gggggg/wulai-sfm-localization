#!/usr/bin/env bash
# Explicitly permit exactly one Sphinx-only probe during the next boot.
set -Eeuo pipefail

umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${ANAFI_PCMD_SIM_PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
ARM_FILE="$PROJECT_DIR/.boot-probe.armed"
SERVICE_NAME="anafi-pcmd-sim-next-boot.service"

if [[ -e "$ARM_FILE" ]]; then
  printf 'boot probe is already armed: %s\n' "$ARM_FILE" >&2
  exit 1
fi

printf '{"armed_at_utc":"%s","service":"%s"}\n' \
  "$(date --utc --iso-8601=seconds)" "$SERVICE_NAME" > "$ARM_FILE"
systemctl --user reset-failed "$SERVICE_NAME" || true
printf 'boot probe armed: %s\n' "$ARM_FILE"
