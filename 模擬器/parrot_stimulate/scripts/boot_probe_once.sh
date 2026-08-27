#!/usr/bin/env bash
# Run exactly one already-armed simulator-only probe after a user manager starts.
set -Eeuo pipefail

umask 077
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${ANAFI_PCMD_SIM_PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
ARM_FILE="$PROJECT_DIR/.boot-probe.armed"
ARTIFACT_ROOT="$PROJECT_DIR/artifacts/boot-runs"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

log() {
  printf '%s %s\n' "$(date --utc --iso-8601=seconds)" "$*"
}

wait_for_firmwared() {
  local deadline=$((SECONDS + 120))
  while ! systemctl is-active --quiet firmwared.service; do
    if (( SECONDS >= deadline )); then
      log "firmwared.service was not active within 120 seconds"
      return 1
    fi
    sleep 2
  done
}

if [[ ! -f "$ARM_FILE" ]]; then
  log "no arm marker present; refusing to run"
  exit 0
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  log "project directory is unavailable: $PROJECT_DIR"
  exit 2
fi

if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  log "uv executable is unavailable: $UV_BIN"
  exit 2
fi

mkdir -p -- "$ARTIFACT_ROOT"
wait_for_firmwared

log "running Sphinx preflight before consuming the arm marker"
if ! "$UV_BIN" run --locked --offline --no-sync anafi-pcmd-sim preflight; then
  log "preflight failed; arm marker remains for a later boot"
  exit 2
fi

RUN_ID="$(date --utc +%Y%m%dT%H%M%SZ)-boot"
OUTPUT_DIR="$ARTIFACT_ROOT/$RUN_ID"
CONSUMED_ARM_FILE="$PROJECT_DIR/.boot-probe.consumed-$RUN_ID"

# A passing preflight consumes the one-shot arm before simulator processes launch.
# This prevents an unexplained flight-probe failure from being retried indefinitely at boot.
mv -- "$ARM_FILE" "$CONSUMED_ARM_FILE"
log "starting one Sphinx-only diagonal probe; output=$OUTPUT_DIR"

exec "$UV_BIN" run --locked --offline --no-sync anafi-pcmd-sim run \
  --launch-sphinx --duration 1.5 --pitch 20 --gaz 20 --output-dir "$OUTPUT_DIR"
