#!/usr/bin/env bash
# Stage one known site into the actual portable export, install its pinned runtime,
# launch its selector/UI, require a valid pose, then restore the export boundary.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: bash tools/test_portable_runtime.sh /path/to/portable-package" >&2
  exit 2
fi

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
portable_root="$(realpath -e -- "$1")"
temporary_root="$(mktemp -d -t sfm-portable-runtime-XXXXXX)"
staged_map_root="$portable_root/地圖檔"
staged_video_root="$portable_root/模擬器/測試影片"
staged_output_root="$portable_root/outputs"
staging_owned=0

cleanup() {
  if [[ "$staging_owned" == "1" ]] && [[ -d "$staged_map_root" ]]; then
    rm -r -- "$staged_map_root"
  fi
  if [[ "$staging_owned" == "1" ]] && [[ -d "$staged_video_root" ]]; then
    rm -r -- "$staged_video_root"
  fi
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
for path in "$staged_map_root" "$staged_video_root" "$portable_root/.venv"; do
  if [[ -e "$path" ]]; then
    echo "[portable-runtime] refusing to overwrite existing path: $path" >&2
    exit 2
  fi
done
for path in \
  "$portable_root/tools/install_runtime.sh" \
  "$portable_root/tools/simulated_ui_smoke.sh" \
  "$portable_root/outputs/README.md" \
  "$portable_root/控制介面程式/site_profiles/river_site_edm.json"; do
  if [[ ! -f "$path" ]]; then
    echo "[portable-runtime] portable package is missing: $path" >&2
    exit 1
  fi
done
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

staging_owned=1
install -d \
  "$portable_root/地圖檔/場域/river_site/maps" \
  "$portable_root/地圖檔/場域/river_site/bundles" \
  "$portable_root/地圖檔/場域/river_site/routes/authored" \
  "$portable_root/地圖檔/場域/urai/reports" \
  "$staged_video_root"
install -m 0644 \
  "$source_root/地圖檔/場域/river_site/maps/river_site_realrgb_dense_trimmed.ply" \
  "$portable_root/地圖檔/場域/river_site/maps/river_site_realrgb_dense_trimmed.ply"
install -m 0644 \
  "$source_root/地圖檔/場域/river_site/maps/river_site_ref_poses.json" \
  "$portable_root/地圖檔/場域/river_site/maps/river_site_ref_poses.json"
install -m 0644 \
  "$source_root/地圖檔/場域/river_site/maps/T_align_gravity.json" \
  "$portable_root/地圖檔/場域/river_site/maps/T_align_gravity.json"
install -m 0644 \
  "$source_root/地圖檔/場域/river_site/bundles/river_site_reloc_map_edm.pt" \
  "$portable_root/地圖檔/場域/river_site/bundles/river_site_reloc_map_edm.pt"
install -m 0644 \
  "$source_root/地圖檔/場域/river_site/routes/authored/route_20260807_013811.json" \
  "$portable_root/地圖檔/場域/river_site/routes/authored/route_20260807_013811.json"
install -m 0644 \
  "$source_root/地圖檔/場域/urai/reports/hardware_approval_20260806.json" \
  "$portable_root/地圖檔/場域/urai/reports/hardware_approval_20260806.json"
install -m 0644 \
  "$source_root/模擬器/測試影片/河濱_P1180118_first_2s.mp4" \
  "$staged_video_root/河濱_P1180118_first_2s.mp4"

portable_venv="$temporary_root/venv"
SFM_VENV_DIR="$portable_venv" \
  bash "$portable_root/tools/install_runtime.sh"
SFM_UI_PYTHON="$portable_venv/bin/python" \
SFM_LOCALIZER_PYTHON="$portable_venv/bin/python" \
SFM_PORTABLE_ALLOW_EXTERNAL_PYTHON=1 \
  bash "$portable_root/tools/simulated_ui_smoke.sh"

echo "[portable-runtime] PASS: actual portable clean install and valid pose"
