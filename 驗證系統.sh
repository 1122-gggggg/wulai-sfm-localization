#!/usr/bin/env bash
set -euo pipefail

validation_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$validation_root/.venv/bin/python" "$validation_root/tools/system_validation.py" "$@"
