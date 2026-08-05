#!/usr/bin/env bash
# Cancel an armed boot probe without disabling the installed service.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${ANAFI_PCMD_SIM_PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
ARM_FILE="$PROJECT_DIR/.boot-probe.armed"

if [[ -e "$ARM_FILE" ]]; then
  rm -f -- "$ARM_FILE"
  printf 'boot probe disarmed\n'
else
  printf 'no armed boot probe was present\n'
fi
