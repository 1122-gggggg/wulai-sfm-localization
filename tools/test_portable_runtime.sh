#!/usr/bin/env bash
# Install an exported site's pinned runtime, launch its selector/UI against an
# explicit external replay, require a valid pose, then remove test outputs.
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 || ( "$#" -eq 2 && "$2" != "--require-site-bundle" ) ]]; then
  echo "usage: bash tools/test_portable_runtime.sh /path/to/portable-package [--require-site-bundle]" >&2
  exit 2
fi

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
portable_root="$(realpath -e -- "$1")"
temporary_root="$(mktemp -d -t sfm-portable-runtime-XXXXXX)"
staged_output_root="$portable_root/outputs"
smoke_video="${SFM_SMOKE_VIDEO:-${SFM_P119_VIDEO:-$source_root/模擬器/測試影片/720p/P1190119_720p.MP4}}"
staging_owned=0

cleanup() {
  if [[ "$staging_owned" == "1" ]] && [[ -d "$staged_output_root/flight_logs" ]]; then
    rm -r -- "$staged_output_root/flight_logs"
  fi
  if [[ -d "$temporary_root" ]]; then
    rm -r -- "$temporary_root"
  fi
}
trap cleanup EXIT

if [[ "$portable_root" == "$source_root" ]]; then
  echo "[portable-runtime] package must differ from the source workspace" >&2
  exit 2
fi
for path in "$staged_output_root"; do
  if [[ -L "$path" ]]; then
    echo "[portable-runtime] refusing symlinked package path: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$staged_output_root" ]]; then
  echo "[portable-runtime] package outputs directory is missing: $staged_output_root" >&2
  exit 2
fi

for path in "$portable_root/.venv"; do
  if [[ -e "$path" || -L "$path" ]]; then
    echo "[portable-runtime] refusing to overwrite existing path: $path" >&2
    exit 2
  fi
done
if [[ ! -f "$portable_root/PORTABLE_PACKAGE.json" ]]; then
  echo "[portable-runtime] portable package is missing: $portable_root/PORTABLE_PACKAGE.json" >&2
  exit 1
fi

package_meta="$("$source_root/.venv/bin/python" - "$portable_root/PORTABLE_PACKAGE.json" <<'PY'
import json
import sys
from pathlib import Path

try:
    raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    metadata = json.loads(raw)
except (json.JSONDecodeError, OSError, UnicodeError) as exc:
    raise SystemExit(f"[portable-runtime] malformed PORTABLE_PACKAGE.json: {exc}")

if not isinstance(metadata, dict):
    raise SystemExit("[portable-runtime] malformed PORTABLE_PACKAGE.json: root must be a JSON object")

package_kind = metadata.get("package_kind")
schema = metadata.get("schema")
entrypoint = metadata.get("entrypoint")

if not isinstance(package_kind, str):
    raise SystemExit("[portable-runtime] malformed PORTABLE_PACKAGE.json: package_kind must be a string")

SUPPORTED_PACKAGES = {
    "live-operator-runtime": {
        "schema": "sfm-portable-live-runtime/v1",
        "entrypoint": "一鍵啟動.sh",
    },
    "simulated-interface-runtime": {
        "schema": "sfm-portable-simulator/v2",
        "entrypoint": "控制介面程式/影片模擬串流/選擇啟動.sh",
    },
}

if package_kind not in SUPPORTED_PACKAGES:
    raise SystemExit(f"[portable-runtime] unknown package_kind: {package_kind!r}")

expected = SUPPORTED_PACKAGES[package_kind]
if not isinstance(schema, str) or schema != expected["schema"]:
    raise SystemExit(
        f"[portable-runtime] schema mismatch for {package_kind}: "
        f"expected {expected['schema']!r}, got {schema!r}"
    )
if not isinstance(entrypoint, str) or entrypoint != expected["entrypoint"]:
    raise SystemExit(
        f"[portable-runtime] entrypoint mismatch for {package_kind}: "
        f"expected {expected['entrypoint']!r}, got {entrypoint!r}"
    )

offline_install = metadata.get("offline_install")
if not isinstance(offline_install, dict) or offline_install.get("complete") is not True:
    raise SystemExit(
        "[portable-runtime] PORTABLE_PACKAGE.json must set "
        "offline_install.complete to literal true"
    )

print(f"{package_kind}\t{entrypoint}")
PY
)"

package_kind="${package_meta%%	*}"
package_entrypoint="${package_meta#*	}"

for path in \
  "$portable_root/PORTABLE_PACKAGE.json" \
  "$portable_root/$package_entrypoint" \
  "$portable_root/tools/install_runtime.sh" \
  "$portable_root/tools/simulated_ui_smoke.sh" \
  "$portable_root/執行環境/offline_wheelhouse/WHEELHOUSE.json" \
  "$portable_root/outputs/README.md"; do
  if [[ ! -f "$path" ]]; then
    echo "[portable-runtime] portable package is missing: $path" >&2
    exit 1
  fi
done

has_site=0
if [[ -f "$portable_root/PORTABLE_SITE_ASSETS.json" ]]; then
  has_site=1
elif [[ "${2:-}" == "--require-site-bundle" ]]; then
  echo "[portable-runtime] localization verification requires a site bundle; export with --site-profile" >&2
  exit 2
fi

if [[ "$has_site" == "1" && ! -s "$smoke_video" ]]; then
  echo "[portable-runtime] smoke video is missing or empty: $smoke_video" >&2
  exit 2
fi


offline_wheelhouse="$portable_root/執行環境/offline_wheelhouse"
wheelhouse_payload="$(find "$offline_wheelhouse" -type f ! -name WHEELHOUSE.json -print -quit)"
if [[ -z "$wheelhouse_payload" ]]; then
  echo "[portable-runtime] offline wheelhouse has no package files: $offline_wheelhouse" >&2
  exit 1
fi
unexpected_outputs=()
while IFS= read -r -d '' path; do
  unexpected_outputs+=("$path")
done < <(
  find "$staged_output_root" -mindepth 1 -maxdepth 1 \
    ! -name README.md -print0
)
if ((${#unexpected_outputs[@]})); then
  printf '[portable-runtime] refusing package with pre-existing output: %s\n' \
    "${unexpected_outputs[@]}" >&2
  exit 2
fi

"$source_root/.venv/bin/python" \
  "$portable_root/tools/package_manifest.py" verify --root "$portable_root"

portable_venv="$temporary_root/venv"
install_log="$temporary_root/offline-install.log"
if ! PIP_NO_INDEX=1 \
  PIP_INDEX_URL=http://127.0.0.1:9/invalid \
  PIP_EXTRA_INDEX_URL= \
  SFM_VENV_DIR="$portable_venv" \
  bash "$portable_root/tools/install_runtime.sh" --offline 2>&1 | tee "$install_log"; then
  echo "[portable-runtime] offline installer failed" >&2
  exit 1
fi
if grep -Eiq 'https?://|looking in indexes:' "$install_log"; then
  echo "[portable-runtime] offline installer attempted a package index or URL" >&2
  exit 1
fi
if [[ "$has_site" == "0" ]]; then
  SFM_WORKSPACE_ROOT="$portable_root" "$portable_venv/bin/python" - "$portable_root" <<'PYTHON'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "控制介面程式/operator_interface"))
import flight_operator_app
import live_localizer_worker
print("[portable-runtime] PASS: runtime-only package imported; no site localization or GUI pose was evaluated")
PYTHON
  exit 0
fi
staging_owned=1
SFM_UI_PYTHON="$portable_venv/bin/python" \
SFM_LOCALIZER_PYTHON="$portable_venv/bin/python" \
SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON=1 \
SFM_SMOKE_VIDEO="$smoke_video" \
  bash "$portable_root/tools/simulated_ui_smoke.sh"
if [[ "$package_kind" == "live-operator-runtime" ]]; then
  SFM_UI_PYTHON="$portable_venv/bin/python" \
  SFM_LOCALIZER_PYTHON="$portable_venv/bin/python" \
  SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON=1 \
  SFM_LAUNCH_DRY_RUN=1 \
  SFM_EVALUATION_ONLY=1 \
  SFM_MAX_PERFORMANCE=0 \
    bash "$portable_root/一鍵啟動.sh"

  echo "[portable-runtime] PASS: clean install, valid pose, and live launcher dry-run"
else
  echo "[portable-runtime] PASS: clean install, valid pose, and simulated interface smoke"
fi
