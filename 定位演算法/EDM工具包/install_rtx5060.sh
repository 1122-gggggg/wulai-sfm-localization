#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${script_dir}/config.json" ]]; then
  package_root="${script_dir}"
else
  package_root="$(cd -- "${script_dir}/.." && pwd)"
fi
venv_dir="${1:-${package_root}/.venv}"
python_bin="${EDM_PYTHON:-python3}"

"${python_bin}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip
"${venv_dir}/bin/python" -m pip install \
  "torch>=2.7" torchvision \
  --index-url https://download.pytorch.org/whl/cu128
"${venv_dir}/bin/python" -m pip install -r "${package_root}/requirements.txt"

cd -- "${package_root}"
"${venv_dir}/bin/python" tests/verify_package.py \
  --require-rtx5060 \
  --max-vram-mib "${EDM_MAX_VRAM_MIB:-7600}" \
  --report rtx5060_preflight.json
