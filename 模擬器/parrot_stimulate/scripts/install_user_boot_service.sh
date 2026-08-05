#!/usr/bin/env bash
# Install and enable the per-user boot service. No root access is required.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
UNIT_NAME="anafi-pcmd-sim-next-boot.service"
UNIT_SOURCE="$PROJECT_DIR/systemd/user/$UNIT_NAME"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT_DESTINATION="$UNIT_DIR/$UNIT_NAME"

if [[ "$PROJECT_DIR" =~ [[:space:]] ]]; then
  printf 'project directory must not contain whitespace: %s\n' "$PROJECT_DIR" >&2
  exit 2
fi

if [[ ! -f "$UNIT_SOURCE" ]]; then
  printf 'service template is unavailable: %s\n' "$UNIT_SOURCE" >&2
  exit 2
fi

install -d -m 0700 -- "$UNIT_DIR"
TEMPORARY_UNIT="$(mktemp "$UNIT_DIR/.${UNIT_NAME}.XXXXXX")"
trap 'rm -f -- "$TEMPORARY_UNIT"' EXIT
ESCAPED_PROJECT_DIR="${PROJECT_DIR//\\/\\\\}"
ESCAPED_PROJECT_DIR="${ESCAPED_PROJECT_DIR//&/\\&}"
ESCAPED_PROJECT_DIR="${ESCAPED_PROJECT_DIR//|/\\|}"
sed "s|@PROJECT_DIR@|$ESCAPED_PROJECT_DIR|g" "$UNIT_SOURCE" > "$TEMPORARY_UNIT"
if grep -Fq '@PROJECT_DIR@' "$TEMPORARY_UNIT"; then
  printf 'unresolved project path template in %s\n' "$UNIT_SOURCE" >&2
  exit 2
fi
chmod 0644 -- "$TEMPORARY_UNIT"
mv -f -- "$TEMPORARY_UNIT" "$UNIT_DESTINATION"
trap - EXIT
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
printf 'installed and enabled user service: %s\n' "$UNIT_NAME"
printf 'installed project directory: %s\n' "$PROJECT_DIR"
