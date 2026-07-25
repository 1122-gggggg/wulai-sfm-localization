#!/usr/bin/env bash
# 在 Blender 上為某個場域畫飛行航線。
#
# 用法:
#   ./航線編修.sh [site profile 路徑]
#   預設 site profile: ../site_profiles/urai_edm.json
#
# 流程:
#   1. 開啟 Blender，載入該場域的點雲，鏡頭從正上方俯視
#   2. 側邊欄（按 N）選「航線」分頁 -> 點「點選航點」
#   3. 左鍵在地圖上點航點，會自動連成一條航線；Z 復原上一點；ENTER 完成
#   4. 轉到側面視角，選中某個航點，按 G 再按 Z 調整高度；按「更新航線」重畫
#   5. 按「匯出航線」-> 寫入該場域 routes/ 下的 flight_path.json / .ply
#
# 視角操作（依需求設定，Blender 預設的中鍵操作仍可用）:
#   Ctrl + 左鍵         旋轉
#   Ctrl + Shift + 左鍵 平移
#   滾輪                縮放
set -euo pipefail

AUTHOR_DIR="$(cd "$(dirname "$0")" && pwd)"
CTRL_DIR="$(cd "$AUTHOR_DIR/.." && pwd)"
ROOT="$(cd "$CTRL_DIR/.." && pwd)"

SITE_PROFILE="${1:-${SFM_SITE_PROFILE:-$CTRL_DIR/site_profiles/urai_edm.json}}"
BLENDER="${SFM_BLENDER:-$HOME/.local/bin/blender}"
PY="${SFM_UI_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -f "$SITE_PROFILE" ]]; then
  echo "[航線編修] 找不到 site profile: $SITE_PROFILE" >&2
  exit 1
fi
if [[ ! -x "$BLENDER" ]]; then
  echo "[航線編修] Blender 不可用: $BLENDER" >&2
  echo "[航線編修] 設 SFM_BLENDER=/path/to/blender 或安裝到 ~/.local/bin/blender" >&2
  exit 1
fi

# 從 site profile 取出這個場域的點雲與航線輸出目錄，避免手動指定而配錯場域。
eval "$("$PY" - "$SITE_PROFILE" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
src = Path(sys.argv[1]).resolve()
raw = json.loads(src.read_text(encoding="utf-8"))
base = src.parent


def resolve(value):
    if not value:
        return ""
    p = Path(value).expanduser()
    return str(p if p.is_absolute() else (base / p).resolve())


assets = raw.get("assets") or {}
map_ply = resolve(assets.get("map_ply"))
route = assets.get("route_json")
if route:
    out_dir = str(Path(resolve(route)).parent)
else:
    # route_json 尚未建立：預設寫到場域包的 routes/authored/
    pack = Path(map_ply).parent
    while pack.name and pack.name != "maps":
        pack = pack.parent
    out_dir = str(pack.parent / "routes" / "authored")
print(f"SITE_ID={raw.get('site_id','')!r}")
print(f"SITE_NAME={raw.get('display_name','')!r}")
print(f"MAP_PLY={map_ply!r}")
print(f"REF_POSES={resolve(raw.get('map_reference_poses'))!r}")
print(f"OUT_DIR={out_dir!r}")
PY
)"

if [[ ! -f "$MAP_PLY" ]]; then
  echo "[航線編修] 點雲不存在: $MAP_PLY" >&2
  exit 1
fi

export SFM_MAP_PLY="$MAP_PLY"
export SFM_MAP_REFERENCE_POSES="$REF_POSES"
export SFM_SAFEZONE_DIR="$OUT_DIR"
export SFM_ROUTE_TOOLS="$AUTHOR_DIR/route_click_tools.py"
export DISPLAY="${DISPLAY:-:0}"

mkdir -p "$OUT_DIR"
echo "[航線編修] 場域：$SITE_NAME ($SITE_ID)"
echo "[航線編修] 點雲：$MAP_PLY"
echo "[航線編修] 輸出：$OUT_DIR/flight_path.json"
echo "[航線編修] 視角：Ctrl+左鍵旋轉、Ctrl+Shift+左鍵平移、滾輪縮放"
echo "[航線編修] 按 N 開側邊欄 -> 「航線」分頁 -> 點選航點；調高度用 G 再 Z"
echo
exec "$BLENDER" --python "$AUTHOR_DIR/blender_route_setup.py"
