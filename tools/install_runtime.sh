#!/usr/bin/env bash
set -euo pipefail

install_test_deps=0
install_quality_deps=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --test-deps)
      install_test_deps=1
      shift
      ;;
    --quality-deps)
      install_quality_deps=1
      shift
      ;;
    --help|-h)
      break
      ;;
    *)
      echo "[runtime] 未知選項: $1" >&2
      exit 2
      ;;
  esac
done
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法: bash tools/install_runtime.sh

環境變數:
  SFM_PYTHON=python3.10       CPython 3.10 executable
  SFM_VENV_DIR=/tmp/sfm-venv  alternate venv directory for clean-install checks
  SFM_INSTALL_TEST_DEPS=1     additionally install requirements-test-lock.txt
  SFM_INSTALL_QUALITY_DEPS=1  additionally install requirements-quality-lock.txt

選項:
  --test-deps                  同上，安裝 hash-locked pytest/ruff/coverage tools
  --quality-deps               安裝 hash-locked mypy/pip-audit/SBOM tools
EOF
  exit 0
fi

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SFM_PYTHON:-python3.10}"
venv_dir="${SFM_VENV_DIR:-$root_dir/.venv}"
venv_python="$venv_dir/bin/python"
requirements_lock="$root_dir/requirements-lock.txt"
requirements_test_lock="$root_dir/requirements-test-lock.txt"
requirements_quality_lock="$root_dir/requirements-quality-lock.txt"
if [[ "${SFM_INSTALL_TEST_DEPS:-0}" == "1" ]]; then
  install_test_deps=1
fi
if [[ "${SFM_INSTALL_QUALITY_DEPS:-0}" == "1" ]]; then
  install_quality_deps=1
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "[runtime] 找不到 $python_bin；需要 CPython 3.10" >&2
  exit 1
fi

if [[ ! -x "$venv_python" ]]; then
  echo "[runtime] 建立 $venv_dir"
  if ! "$python_bin" -m venv "$venv_dir"; then
    echo "[runtime] 無法建立 venv；請安裝 CPython 3.10 的 venv/ensurepip 支援" >&2
    echo "[runtime] Ubuntu/Debian 通常需要套件：python3.10-venv" >&2
    exit 1
  fi
fi

if [[ -f "$venv_dir/pyvenv.cfg" ]] && grep -qi '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "$venv_dir/pyvenv.cfg"; then
  echo "[runtime] 拒絕使用 include-system-site-packages=true 的環境: $venv_dir" >&2
  echo "[runtime] 請以乾淨目錄設定 SFM_VENV_DIR，或重新建立此 venv（不得使用系統套件）" >&2
  exit 1
fi

"$venv_python" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"[runtime] .venv 必須是 CPython 3.10，實際為 {sys.version.split()[0]}"
    )
PY

if [[ ! -f "$requirements_lock" ]]; then
  echo "[runtime] 缺少固定相依鎖檔: $requirements_lock" >&2
  exit 1
fi
"$venv_python" -m pip install --require-hashes --requirement "$requirements_lock"
if [[ "$install_test_deps" == "1" ]]; then
  if [[ ! -f "$requirements_test_lock" ]]; then
    echo "[runtime] 缺少固定測試相依鎖檔: $requirements_test_lock" >&2
    exit 1
  fi
  "$venv_python" -m pip install --require-hashes --requirement "$requirements_test_lock"
fi
if [[ "$install_quality_deps" == "1" ]]; then
  if [[ ! -f "$requirements_quality_lock" ]]; then
    echo "[runtime] 缺少固定品質相依鎖檔: $requirements_quality_lock" >&2
    exit 1
  fi
  "$venv_python" -m pip install --require-hashes --requirement "$requirements_quality_lock"
fi
echo "[runtime] 安裝完成：$venv_python"
echo "[runtime] 系統層仍需 ffmpeg、python3-tk、X11/XWayland、可用 NVIDIA CUDA driver"
