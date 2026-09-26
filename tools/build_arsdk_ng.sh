#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${SFM_ARSDK_NG_LIB_DIR:-$ROOT/執行環境/native/arsdk-ng/lib}"
JOBS="${SFM_ARSDK_NG_JOBS:-$(nproc)}"
SOURCE_ROOT="${SFM_ARSDK_NG_SOURCE_ROOT:-}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法: bash tools/build_arsdk_ng.sh

從 Parrot 官方 arsdk-ng 與固定 commit 的 Alchemy、arsdk-xml、libpomp、
libfutils、libmux、ulog 建置 x86_64 Linux shared libraries。
輸出預設為 執行環境/native/arsdk-ng/lib；真機 launcher 會自動載入。

環境變數:
  SFM_ARSDK_NG_LIB_DIR       shared library 輸出目錄
  SFM_ARSDK_NG_SOURCE_ROOT   已 checkout 的官方 source workspace（各 repo 必須在鎖定 commit）
  SFM_ARSDK_NG_JOBS          make 平行工作數
EOF
  exit 0
fi

if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[arsdk-ng] ERROR: SFM_ARSDK_NG_JOBS must be a positive integer" >&2
  exit 2
fi

# Commit pins were built and ABI-checked against the direct ctypes bindings.
repos=(
  "alchemy|https://github.com/Parrot-Developers/alchemy.git|d69e22c2dcfa3ec5231fce8686b65e33ce69309d"
  "arsdk-ng|https://github.com/Parrot-Developers/arsdk-ng.git|409daec0e8fc3393c31a8d09505a47d056eaafe9"
  "arsdk-xml|https://github.com/Parrot-Developers/arsdk-xml.git|2299c689be5951fa2104793e0154b0b5b1896c03"
  "libpomp|https://github.com/Parrot-Developers/libpomp.git|7261ed26d668f7b1a26ae4b7baa2d5490f99ce53"
  "libfutils|https://github.com/Parrot-Developers/libfutils.git|aeac5bbd3190bb0ba89cf3e3ca748d2e0a1e1561"
  "libmux|https://github.com/Parrot-Developers/libmux.git|b01f2311db26ee7a31e5053087de32ebfb9f3ca9"
  "ulog|https://github.com/Parrot-Developers/ulog.git|ae3cbbfc6eacb6a9b2f4e1787cc3eeb8afdc2c82"
)

work_root="$(mktemp -d "${TMPDIR:-/tmp}/sfm-arsdk-ng.XXXXXX")"
cleanup() { rm -rf -- "$work_root"; }
trap cleanup EXIT

if [[ -n "$SOURCE_ROOT" ]]; then
  source_root="$(cd "$SOURCE_ROOT" && pwd)"
  for item in "${repos[@]}"; do
    IFS='|' read -r name _ revision <<<"$item"
    checkout="$source_root/$name"
    if [[ ! -d "$checkout/.git" ]]; then
      echo "[arsdk-ng] ERROR: missing Git checkout: $checkout" >&2
      exit 2
    fi
    actual="$(git -C "$checkout" rev-parse HEAD)"
    if [[ "$actual" != "$revision" ]]; then
      echo "[arsdk-ng] ERROR: $name revision $actual; expected $revision" >&2
      exit 2
    fi
  done
else
  source_root="$work_root/source"
  mkdir -p "$source_root"
  for item in "${repos[@]}"; do
    IFS='|' read -r name url revision <<<"$item"
    checkout="$source_root/$name"
    echo "[arsdk-ng] fetching $name@$revision"
    git clone --quiet --depth 1 --filter=blob:none --no-checkout "$url" "$checkout"
    git -C "$checkout" fetch --quiet --depth=1 origin "$revision"
    git -C "$checkout" checkout --quiet --detach FETCH_HEAD
    actual="$(git -C "$checkout" rev-parse HEAD)"
    if [[ "$actual" != "$revision" ]]; then
      echo "[arsdk-ng] ERROR: $name resolved to unexpected revision $actual" >&2
      exit 2
    fi
  done
fi

build_out="$work_root/alchemy-out"
echo "[arsdk-ng] building official libraries ($JOBS jobs)"
(
  cd "$source_root"
  ALCHEMY_TARGET_OUT="$build_out" make -f alchemy/main.mk libarsdk libarsdkctrl -j"$JOBS"
)

staged="$build_out/staging/usr/lib"
required=(libarsdk.so libarsdkctrl.so libpomp.so libfutils.so libmux.so libulog.so)
for library in "${required[@]}"; do
  if [[ ! -f "$staged/$library" ]]; then
    echo "[arsdk-ng] ERROR: build did not produce $staged/$library" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUTPUT_DIR")"
if [[ -L "$OUTPUT_DIR" ]]; then
  echo "[arsdk-ng] ERROR: refusing symlink output directory: $OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"
for library in "${required[@]}"; do
  install -m 755 "$staged/$library" "$OUTPUT_DIR/$library.tmp.$$"
  mv -f -- "$OUTPUT_DIR/$library.tmp.$$" "$OUTPUT_DIR/$library"
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "[arsdk-ng] ERROR: python3 is required for native-library smoke verification" >&2
  exit 1
fi
LD_LIBRARY_PATH="$OUTPUT_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  ARSDK_NG_LIB_DIR="$OUTPUT_DIR" \
  python3 -c 'import ctypes, os; d=os.environ["ARSDK_NG_LIB_DIR"]; [ctypes.CDLL(os.path.join(d, n)) for n in ("libarsdk.so", "libarsdkctrl.so", "libpomp.so", "libfutils.so", "libmux.so", "libulog.so")]'

if [[ -f "$ROOT/PORTABLE_PACKAGE.json" \
  && "$OUTPUT_DIR" == "$ROOT/執行環境/native/arsdk-ng/lib" ]]; then
  echo "[arsdk-ng] refreshing portable package checksums"
  python3 "$ROOT/tools/package_manifest.py" generate --root "$ROOT"
  python3 "$ROOT/tools/package_manifest.py" verify --root "$ROOT"
fi

echo "[arsdk-ng] official libraries installed in $OUTPUT_DIR"
