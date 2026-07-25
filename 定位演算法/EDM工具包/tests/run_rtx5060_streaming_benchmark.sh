#!/usr/bin/env bash
set -euo pipefail

profile="${1:-balanced}"
output_root="${2:-/media/cihcilab/新增磁碟區/sfm_system/建圖/target_site/runs/target_site_v1/edm_streaming_benchmark_rtx5060}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
package_root="$(cd -- "${script_dir}/.." && pwd)"
if [[ -x "${package_root}/env/.venv_edm/bin/python" ]]; then
  python_bin="${package_root}/env/.venv_edm/bin/python"
else
  python_bin="${EDM_PYTHON:-python3}"
fi
benchmark="${script_dir}/bench_streaming_corpus_edm.py"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"

if [[ "${gpu_name}" != *"RTX 5060"* && "${EDM_ALLOW_NON_5060:-0}" != "1" ]]; then
  echo "Refusing to label this an RTX 5060 benchmark: detected '${gpu_name}'." >&2
  echo "Set EDM_ALLOW_NON_5060=1 only for a deliberately labelled non-5060 dry run." >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${output_root}/${profile}_${timestamp}"
common=(
  "${python_bin}" "${benchmark}"
  --out-dir "${out_dir}"
  --progress-every 600
)
if [[ -f "${package_root}/config.json" ]]; then
  common+=(--config "${package_root}/config.json")
fi

case "${profile}" in
  balanced)
    "${common[@]}" --local-topk 1 --edm-topk 2304 --max-corr-total 900
    ;;
  precision)
    "${common[@]}" --local-topk 2 --max-corr-total 1200
    ;;
  *)
    echo "Usage: $0 [balanced|precision] [output_root]" >&2
    exit 2
    ;;
esac

echo "RTX 5060 benchmark result: ${out_dir}/summary.json"
